//! The audio-side runtime: note lifecycle, ADSR/LFO processing, the DS master clock, and the
//! parallel look-ahead [`FsVisController`] used to drive visualizers.

use std::sync::Arc;

use crate::bank::{InstrumentBank, InstrumentType};
use crate::sample::{decode_adpcm, decode_pcm16, decode_pcm8, ResampleMode, Sample};
use crate::sdat::Sdat;
use crate::sequence::{Message, MessageType, Sequence};
use crate::tables::{snd_sin_idx, DECIBEL_SQUARE_TABLE, GET_VOL_TABLE, SQUARE_WAVES};
use crate::tuning::{midi_note_to_hz, TuningSystem};
use crate::util::{read_u16, read_u32, read_u8};
use crate::{SampleSynthesizer, CYCLES_PER_TICK, DS_CLOCK_RATE, TRACK_COUNT};

/// Runtime-tunable synthesis options (replaces the original engine's global flags).
#[derive(Debug, Clone)]
pub struct SynthConfig {
    /// Apply the Haas-effect stereo widening delay lines.
    pub stereo_separation: bool,
    /// Force minimum stereo separation on barely-panned channels.
    pub force_stereo_separation: bool,
    /// Keep low frequencies centered ("bass mono"): only content above
    /// [`Self::bass_mono_freq`] is widened by the stereo separation, while the bass stays glued
    /// to the center.
    pub bass_mono: bool,
    /// Crossover cutoff (Hz) below which the signal is kept mono when [`Self::bass_mono`] is set.
    pub bass_mono_freq: f64,
    /// The active tuning system.
    pub tuning: TuningSystem,
    /// Which of the 16 tracks are mixed into the output.
    pub track_enables: [bool; TRACK_COUNT],
    /// Sample interpolation / anti-aliasing mode.
    pub resample: ResampleMode,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            stereo_separation: false,
            force_stereo_separation: false,
            bass_mono: false,
            bass_mono_freq: 200.0,
            tuning: TuningSystem::Equal,
            track_enables: [true; TRACK_COUNT],
            resample: ResampleMode::NearestNeighbor,
        }
    }
}

/// ADSR envelope stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdsrState {
    Attack,
    Decay,
    Sustain,
    Release,
}

/// LFO waveform target.
mod lfo_type {
    pub const PITCH: i32 = 0;
    pub const VOLUME: i32 = 1;
    pub const PAN: i32 = 2;
}

/// Per-note runtime state. All-`Copy` so the tick loop can read/modify a local copy and write
/// it back, sidestepping borrow conflicts with the synthesizers.
#[derive(Debug, Clone, Copy)]
struct ActiveNote {
    track_num: usize,
    midi_note: u8,
    velocity: i32,
    synth_instr_index: usize,
    start_time: u32,
    end_time: u32,
    adsr_state: AdsrState,
    adsr_timer: i32,
    from_keyboard: bool,
    lfo_counter: i32,
    /// Shared LFO delay/phase counter (pokediamond's single `SNDLfo::delayCounter`).
    delay_counter: i32,
    // Resolved instrument coefficients for this note's region.
    f_record: u8,
    attack_coefficient: i32,
    decay_coefficient: i32,
    sustain_level: i32,
    release_coefficient: i32,
}

/// The audio-side runtime that turns an SSEQ + bank into stereo samples.
pub struct Controller {
    sample_rate: f64,
    /// The running sequence interpreter.
    pub sequence: Sequence,
    /// One polyphonic synthesizer per track.
    pub synthesizers: Vec<SampleSynthesizer>,
    instrument_bank: InstrumentBank,
    decoded_sample_archives: Vec<Option<Vec<Arc<Sample>>>>,
    squares: Vec<Arc<Sample>>,
    active_notes: Vec<ActiveNote>,
    /// `notes_on[track][note]` is 1 while a sequence note sounds (drives the visualizer).
    pub notes_on: Vec<[u8; 128]>,
    /// As [`Self::notes_on`] but for live keyboard input.
    pub notes_on_keyboard: Vec<[u8; 128]>,
    /// Count of sequence jumps seen (used by callers to detect loop points).
    pub jumps: u32,
    /// Set when the controller decides the song should fade out.
    pub fading_start: bool,
    /// Which track receives live keyboard input, if any.
    pub active_keyboard_track_num: Option<usize>,
    bpm_timer: u32,
    timer: f64,
}

impl Controller {
    /// Binds sequence `sseq_id` from `sdat` for playback at `sample_rate`.
    ///
    /// Decodes the linked sample archives up front. Returns `None` if the sequence or its bank
    /// is missing.
    pub fn new(sample_rate: f64, sdat: &Sdat, sseq_id: u32) -> Option<Controller> {
        let sseq_info = sdat.sseq_infos.get(sseq_id as usize)?.clone()?;
        let bank_id = sseq_info.bank as usize;
        let bank_info = sdat.sbnk_infos.get(bank_id)?.clone()?;
        let instrument_bank = sdat.instrument_banks.get(bank_id)?.clone()?;

        let sseq_file = sdat.file(sseq_info.file_id)?;
        let sseq_arc: Arc<[u8]> = Arc::from(sseq_file.to_vec());

        // Decode the up-to-four linked sample archives.
        let mut decoded_sample_archives: Vec<Option<Vec<Arc<Sample>>>> = vec![None; 4];
        for (i, &swar_id) in bank_info.swar_id.iter().enumerate() {
            let Some(Some(swar_info)) = sdat.swar_infos.get(swar_id as usize) else {
                continue;
            };
            let Some(swar_file) = sdat.file(swar_info.file_id) else {
                continue;
            };

            let sample_count = read_u32(swar_file, 0x38) as usize;
            let mut archive = Vec::with_capacity(sample_count);
            for j in 0..sample_count {
                let sample_offset = read_u32(swar_file, 0x3C + j * 4) as usize;

                let wav_type = read_u8(swar_file, sample_offset);
                let loop_flag = read_u8(swar_file, sample_offset + 1);
                let sample_rate_hdr = read_u16(swar_file, sample_offset + 2) as f64;
                let swar_loop_offset = read_u16(swar_file, sample_offset + 6) as i64;
                let swar_sample_length = read_u32(swar_file, sample_offset + 8) as i64;

                let data_len = ((swar_loop_offset + swar_sample_length) * 4) as usize;
                let start = sample_offset + 0xC;
                let sample_data = swar_file.get(start..start + data_len).unwrap_or(&[]);

                let (decoded, loop_point) = match wav_type {
                    0 => (decode_pcm8(sample_data), swar_loop_offset * 4),
                    1 => (decode_pcm16(sample_data), swar_loop_offset * 2),
                    2 => (decode_adpcm(sample_data), swar_loop_offset * 8 - 8),
                    _ => (Vec::new(), 0),
                };

                let mut sample =
                    Sample::new(decoded, 440.0, sample_rate_hdr, loop_flag != 0, loop_point);
                sample.sample_length = (swar_sample_length * 4) as usize;
                archive.push(Arc::new(sample));
            }
            decoded_sample_archives[i] = Some(archive);
        }

        // Build the eight PSG square-wave samples.
        let squares = SQUARE_WAVES
            .iter()
            .map(|wave| {
                let mut s = Sample::new(wave.to_vec(), 1.0, 8.0, true, 0);
                s.is_psg_square = true;
                Arc::new(s)
            })
            .collect();

        let data_offset = read_u32(&sseq_arc, 0x18);
        let sequence = Sequence::new(sseq_arc, data_offset, 1024);

        let synthesizers = (0..TRACK_COUNT)
            .map(|_| SampleSynthesizer::new(sample_rate, 16))
            .collect();

        Some(Controller {
            sample_rate,
            sequence,
            synthesizers,
            instrument_bank,
            decoded_sample_archives,
            squares,
            active_notes: Vec::new(),
            notes_on: vec![[0u8; 128]; TRACK_COUNT],
            notes_on_keyboard: vec![[0u8; 128]; TRACK_COUNT],
            jumps: 0,
            fading_start: false,
            active_keyboard_track_num: None,
            bpm_timer: 0,
            timer: 0.0,
        })
    }

    /// The audio sample rate this controller renders at.
    #[inline]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Advances the DS master clock and returns one mixed stereo sample.
    ///
    /// This is the single place where the hardware tick math lives: the DS clock is accumulated
    /// per output sample and the sequence is ticked every `CYCLES_PER_TICK` cycles.
    pub fn next_sample(&mut self, config: &SynthConfig) -> (f32, f32) {
        self.timer += DS_CLOCK_RATE as f64;
        let threshold = CYCLES_PER_TICK as f64 * self.sample_rate;
        while self.timer >= threshold {
            self.timer -= threshold;
            self.tick(config);
        }

        let mut val_l = 0.0;
        let mut val_r = 0.0;
        for i in 0..TRACK_COUNT {
            self.synthesizers[i].next_sample(config);
            if config.track_enables[i] {
                val_l += self.synthesizers[i].val_l;
                val_r += self.synthesizers[i].val_r;
            }
        }
        (val_l as f32, val_r as f32)
    }

    /// Fills `out` with interleaved stereo (L, R, L, R, …) samples.
    pub fn fill(&mut self, out: &mut [f32], config: &SynthConfig) {
        for frame in out.chunks_mut(2) {
            let (l, r) = self.next_sample(config);
            frame[0] = l;
            if frame.len() > 1 {
                frame[1] = r;
            }
        }
    }

    /// Advances note envelopes/LFOs, then ticks the sequence and applies its messages.
    pub fn tick(&mut self, config: &SynthConfig) {
        self.process_active_notes(config);

        self.bpm_timer += self.sequence.tracks[0].bpm;
        while self.bpm_timer >= 240 {
            self.bpm_timer -= 240;

            // Report which tracks still have sounding/releasing channels, so the sequence can
            // honor pokediamond's `noteFinishWait` (stall after a zero-duration note).
            let mut track_has_channels = [false; TRACK_COUNT];
            for note in &self.active_notes {
                if let Some(slot) = track_has_channels.get_mut(note.track_num) {
                    *slot = true;
                }
            }
            self.sequence.tick(&track_has_channels);

            while let Some(msg) = self.sequence.message_buffer.pop() {
                self.handle_message(msg, config);
            }
        }
    }

    /// Runs one ADSR/LFO update pass over the active notes (mirrors the original exactly,
    /// including that at most one finished note is removed per tick).
    fn process_active_notes(&mut self, config: &SynthConfig) {
        let mut index_to_delete: Option<usize> = None;
        let ticks = self.sequence.ticks_elapsed;

        for index in 0..self.active_notes.len() {
            let mut entry = self.active_notes[index];
            let t = entry.track_num;
            let si = entry.synth_instr_index;

            let (instr_start, instr_playing, looping, data_len, sample_t) = {
                let instr = self.synthesizers[t].instr(si);
                (
                    instr.start_time,
                    instr.playing,
                    instr.sample.looping,
                    instr.sample.data.len(),
                    instr.sample_t,
                )
            };

            if instr_start == entry.start_time && instr_playing {
                // Cut voices whose (non-looping) sample has run out.
                if !looping && sample_t > data_len as f64 {
                    index_to_delete = Some(index);
                    self.synthesizers[t].cut_instrument(si);
                }

                // Begin release once the note's scheduled duration elapses.
                if ticks >= entry.end_time
                    && !entry.from_keyboard
                    && entry.adsr_state != AdsrState::Release
                {
                    self.notes_on[t][entry.midi_note as usize] = 0;
                    entry.adsr_state = AdsrState::Release;
                }

                self.apply_lfo(&mut entry, config);
                self.apply_adsr(&mut entry, index, &mut index_to_delete);
            } else {
                index_to_delete = Some(index);
                self.notes_on[t][entry.midi_note as usize] = 0;
            }

            self.active_notes[index] = entry;
        }

        if let Some(i) = index_to_delete {
            self.active_notes.remove(i);
        }
    }

    /// LFO update for one note, ported faithfully (including the DS fixed-point math).
    fn apply_lfo(&mut self, entry: &mut ActiveNote, config: &SynthConfig) {
        let t = entry.track_num;
        let track = &self.sequence.tracks[t];
        let (lfo_depth, lfo_delay, lfo_type, lfo_speed, lfo_range) = (
            track.lfo_depth,
            track.lfo_delay,
            track.lfo_type,
            track.lfo_speed,
            track.lfo_range,
        );

        // Whether the LFO delay has elapsed (the phase advances this tick). pokediamond gates both
        // the value and the phase on a single `delayCounter`; see [`lfo_tick`].
        let delay_elapsed = entry.delay_counter >= lfo_delay;
        let params = LfoParams {
            depth: lfo_depth,
            delay: lfo_delay,
            lfo_type,
            speed: lfo_speed,
            range: lfo_range,
        };
        let lfo_value = lfo_tick(&params, &mut entry.lfo_counter, &mut entry.delay_counter);

        // Pitch modulation is applied only on ticks where the phase advances (delay elapsed),
        // matching the original; the value is in 1/64ths of a semitone.
        if delay_elapsed && lfo_value != 0 && lfo_type == lfo_type::PITCH {
            self.synthesizers[t]
                .instr_mut(entry.synth_instr_index)
                .set_finetune_lfo(lfo_value as f64 / 64.0, config.tuning);
        }
    }

    /// ADSR envelope advance for one note.
    fn apply_adsr(
        &mut self,
        entry: &mut ActiveNote,
        index: usize,
        index_to_delete: &mut Option<usize>,
    ) {
        let t = entry.track_num;
        let si = entry.synth_instr_index;
        match entry.adsr_state {
            AdsrState::Attack => {
                entry.adsr_timer =
                    -(((-entry.attack_coefficient).wrapping_mul(entry.adsr_timer)) >> 8);
                self.synthesizers[t].instr_mut(si).volume =
                    calc_channel_volume(entry.velocity, entry.adsr_timer);
                if entry.adsr_timer == 0 {
                    entry.adsr_state = AdsrState::Decay;
                }
            }
            AdsrState::Decay => {
                entry.adsr_timer -= entry.decay_coefficient;
                if entry.adsr_timer <= entry.sustain_level {
                    entry.adsr_timer = entry.sustain_level;
                    entry.adsr_state = AdsrState::Sustain;
                }
                self.synthesizers[t].instr_mut(si).volume =
                    calc_channel_volume(entry.velocity, entry.adsr_timer);
            }
            AdsrState::Sustain => {
                self.synthesizers[t].instr_mut(si).volume =
                    calc_channel_volume(entry.velocity, entry.adsr_timer);
            }
            AdsrState::Release => {
                if entry.adsr_timer <= -92544 || entry.f_record == 0x2
                /* PSG pulse */
                {
                    self.synthesizers[t].cut_instrument(si);
                    *index_to_delete = Some(index);
                    self.notes_on[t][entry.midi_note as usize] = 0;
                } else {
                    entry.adsr_timer -= entry.release_coefficient;
                    self.synthesizers[t].instr_mut(si).volume =
                        calc_channel_volume(entry.velocity, entry.adsr_timer);
                }
            }
        }
    }

    /// Applies one sequence message.
    fn handle_message(&mut self, msg: Message, config: &SynthConfig) {
        match msg.msg_type {
            MessageType::PlayNote => self.play_note(msg, config),
            MessageType::Jump => self.jumps += 1,
            MessageType::TrackEnded => {
                let any_active = self.sequence.tracks.iter().any(|tr| tr.active);
                if !any_active {
                    self.fading_start = true;
                }
            }
            MessageType::VolumeChange => {
                self.synthesizers[msg.track_num].volume = msg.param0 as f64 / 127.0;
            }
            MessageType::PanChange => {
                self.synthesizers[msg.track_num].set_pan(msg.param0 as f64 / 128.0, config);
            }
            MessageType::PitchBend => {
                let track = &self.sequence.tracks[msg.track_num];
                // `pitch_bend` is already a signed byte (pokediamond `par._s8`, set by `0xC4`);
                // scale by half the bend range, in 1/64 semitones.
                let pitch_bend = track.pitch_bend;
                let finetune = (pitch_bend as f64) * (track.pitch_bend_range as f64 / 2.0) / 64.0;
                self.synthesizers[msg.track_num].set_finetune(finetune, config.tuning);
            }
            MessageType::InstrumentChange => {}
        }
    }

    /// Starts a note from a [`MessageType::PlayNote`] message.
    fn play_note(&mut self, msg: Message, config: &SynthConfig) {
        let t = msg.track_num;
        // The active keyboard track is silenced by the sequence (live input drives it instead).
        if self.active_keyboard_track_num == Some(t) && !msg.from_keyboard {
            return;
        }

        let midi_note = msg.param0;
        let velocity = msg.param1;
        let duration = msg.param2 as u32;
        let program = self.sequence.tracks[t].program;

        // Resolve the region + sample, copying out everything we need before we mutate `self`.
        let Some(instrument) = self.instrument_bank.instruments.get(program) else {
            return;
        };
        let index = instrument.resolve_entry_index(midi_note as u8);
        let Some(region) = instrument.regions.get(index) else {
            return;
        };
        let is_psg_pulse = instrument.instrument_type() == Some(InstrumentType::PsgPulse);
        let archive_index = region.swar_info_id as usize;
        let sample_id = region.swav_info_id as usize;
        let note_number = region.note_number;
        let f_record = instrument.f_record;
        let attack_coefficient = region.attack_coefficient;
        let decay_coefficient = region.decay_coefficient;
        let sustain_level = region.sustain_level;
        let release_coefficient = region.release_coefficient;

        // Pick the sample and the pitch it represents.
        let (sample, sample_frequency) = if is_psg_pulse {
            match self.squares.get(sample_id) {
                Some(s) => (s.clone(), s.frequency),
                None => return,
            }
        } else {
            let Some(Some(archive)) = self.decoded_sample_archives.get(archive_index) else {
                return;
            };
            let Some(sample) = archive.get(sample_id) else {
                return;
            };
            (
                sample.clone(),
                midi_note_to_hz(f64::from(note_number), config.tuning),
            )
        };

        let initial_volume = if attack_coefficient == 0 {
            calc_channel_volume(velocity, 0)
        } else {
            0.0
        };

        let synth_instr_index = self.synthesizers[t].play(
            sample,
            f64::from(midi_note),
            sample_frequency,
            initial_volume,
            self.sequence.ticks_elapsed,
            config.tuning,
        );

        self.notes_on[t][midi_note as usize] = 1;
        self.active_notes.push(ActiveNote {
            track_num: t,
            midi_note: midi_note as u8,
            velocity,
            synth_instr_index,
            start_time: self.sequence.ticks_elapsed,
            end_time: self.sequence.ticks_elapsed + duration,
            adsr_state: AdsrState::Attack,
            adsr_timer: -92544,
            from_keyboard: msg.from_keyboard,
            lfo_counter: 0,
            delay_counter: 0,
            f_record,
            attack_coefficient,
            decay_coefficient,
            sustain_level,
            release_coefficient,
        });
    }

    /// Triggers a note from live keyboard input on `track` (mirrors the legacy app's
    /// `sendMessage(true, PlayNote, ...)` path). The note sounds until released.
    pub fn play_keyboard_note(
        &mut self,
        track: usize,
        note: u8,
        velocity: i32,
        duration: u32,
        config: &SynthConfig,
    ) {
        if track >= TRACK_COUNT {
            return;
        }
        let msg = Message {
            from_keyboard: true,
            track_num: track,
            msg_type: MessageType::PlayNote,
            param0: i32::from(note),
            param1: velocity,
            param2: duration as i32,
            timestamp: self.sequence.ticks_elapsed,
        };
        self.play_note(msg, config);
        self.notes_on_keyboard[track][note as usize] = 1;
    }

    /// Releases a previously-triggered keyboard note, moving it into its ADSR release stage.
    pub fn release_keyboard_note(&mut self, track: usize, note: u8) {
        if track >= TRACK_COUNT {
            return;
        }
        for entry in self.active_notes.iter_mut() {
            if entry.track_num == track && entry.midi_note == note {
                entry.adsr_state = AdsrState::Release;
            }
        }
        self.notes_on_keyboard[track][note as usize] = 0;
    }
}

/// Computes a channel's linear volume from velocity and ADSR timer.
///
/// Based on `SND_CalcChannelVolume` from `pret/pokediamond`.
pub fn calc_channel_volume(velocity: i32, adsr_timer: i32) -> f64 {
    const SND_VOL_DB_MIN: i32 = -723;

    let mut vol = DECIBEL_SQUARE_TABLE[velocity as usize];
    vol += adsr_timer >> 7;
    vol = vol.clamp(SND_VOL_DB_MIN, 0);

    let mut result = f64::from(GET_VOL_TABLE[(vol - SND_VOL_DB_MIN) as usize]);
    if vol < -240 {
        result /= 16.0;
    } else if vol < -120 {
        result /= 4.0;
    } else if vol < -60 {
        result /= 2.0;
    }

    result / 127.0
}

/// LFO parameters for one tick (mirrors the relevant fields of pokediamond's `SNDLfoParam`).
#[derive(Debug)]
struct LfoParams {
    depth: i32,
    delay: i32,
    lfo_type: i32,
    speed: i32,
    range: i32,
}

/// Advances one LFO tick exactly as pokediamond's `SND_GetLfoValue` + `SND_UpdateLfo`.
///
/// A single `delay_counter` gates both the value and the phase: while it is below `delay` the
/// returned value is 0 and the phase (`counter`) is frozen while `delay_counter` counts up; once
/// the delay elapses the value engages and `counter` advances by `speed << 6` per tick. Returns
/// the modulation value after the per-target scaling (`*60` for volume, `<<6` for pitch/pan) and
/// the final `>> 14`.
fn lfo_tick(p: &LfoParams, counter: &mut i32, delay_counter: &mut i32) -> i64 {
    let mut value: i64 = if p.depth == 0 || *delay_counter < p.delay {
        0
    } else {
        i64::from(snd_sin_idx(*counter >> 8)) * i64::from(p.depth) * i64::from(p.range)
    };

    if value != 0 {
        match p.lfo_type {
            lfo_type::VOLUME => value *= 60,
            lfo_type::PITCH | lfo_type::PAN => value <<= 6,
            _ => {}
        }
        value >>= 14;
    }

    if *delay_counter < p.delay {
        *delay_counter += 1;
    } else {
        let mut tmp = *counter;
        tmp += p.speed << 6;
        tmp >>= 8;
        while tmp >= 0x80 {
            tmp -= 0x80;
        }
        *counter += p.speed << 6;
        *counter &= 0xFF;
        *counter |= tmp << 8;
    }

    value
}

/// A pitch-bend change observed by the look-ahead, in resolved semitones at a given tick.
#[derive(Debug, Clone, Copy)]
pub struct PitchBendEvent {
    /// Tick at which the bend took effect.
    pub timestamp: u32,
    /// Track the bend applies to.
    pub track: usize,
    /// Bend amount in semitones (`pitch_bend * range/2 / 64`).
    pub semitones: f32,
}

/// A parallel sequence runner used to drive look-ahead visualizers without producing audio.
///
/// It runs the same SSEQ as [`Controller`] but only tracks which notes are on, and is advanced
/// `run_ahead_ticks` ahead at construction.
pub struct FsVisController {
    /// The look-ahead sequence.
    pub sequence: Sequence,
    /// Recently triggered notes, newest last (capacity-bounded).
    pub active_notes: crate::util::CircularBuffer<Message>,
    /// Recently observed pitch-bend changes, newest last (capacity-bounded).
    pub pitch_bends: crate::util::CircularBuffer<PitchBendEvent>,
    bpm_timer: u32,
}

impl FsVisController {
    /// Builds a look-ahead controller for `sseq_id`, advanced `run_ahead_ticks` ticks.
    pub fn new(sdat: &Sdat, sseq_id: u32, run_ahead_ticks: u32) -> Option<FsVisController> {
        let info = sdat.sseq_infos.get(sseq_id as usize)?.clone()?;
        let file = sdat.file(info.file_id)?;
        let arc: Arc<[u8]> = Arc::from(file.to_vec());
        let data_offset = read_u32(&arc, 0x18);

        let mut ctrl = FsVisController {
            sequence: Sequence::new(arc, data_offset, 512),
            active_notes: crate::util::CircularBuffer::new(2048),
            pitch_bends: crate::util::CircularBuffer::new(2048),
            bpm_timer: 0,
        };
        for _ in 0..run_ahead_ticks {
            ctrl.tick();
        }
        Some(ctrl)
    }

    /// Advances the look-ahead sequence by one tick, recording note-on events.
    pub fn tick(&mut self) {
        self.bpm_timer += self.sequence.tracks[0].bpm;
        while self.bpm_timer >= 240 {
            self.bpm_timer -= 240;
            // The look-ahead visualizer has no channel state; pass all-false so zero-duration
            // notes advance immediately rather than stalling.
            self.sequence.tick(&[false; TRACK_COUNT]);

            while let Some(mut msg) = self.sequence.message_buffer.pop() {
                match msg.msg_type {
                    MessageType::PlayNote => {
                        if self.active_notes.is_full() {
                            self.active_notes.pop();
                        }
                        msg.timestamp = self.sequence.ticks_elapsed;
                        self.active_notes.insert(msg);
                    }
                    MessageType::PitchBend => {
                        // Resolve the current bend in semitones from the track's live state,
                        // matching the audio controller's `set_finetune` math.
                        let tr = &self.sequence.tracks[msg.track_num];
                        let semitones =
                            tr.pitch_bend as f32 * (tr.pitch_bend_range as f32 / 2.0) / 64.0;
                        if self.pitch_bends.is_full() {
                            self.pitch_bends.pop();
                        }
                        self.pitch_bends.insert(PitchBendEvent {
                            timestamp: self.sequence.ticks_elapsed,
                            track: msg.track_num,
                            semitones,
                        });
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation of pokediamond's `SND_GetLfoValue` + `SND_UpdateLfo` for one
    /// tick, used as the oracle for [`lfo_tick`]. Transcribed directly from `SND_exChannel.c`.
    fn pokediamond_lfo(
        p: &LfoParams,
        counter: &mut i32,
        delay_counter: &mut i32,
        target_scale: bool,
    ) -> i64 {
        // SND_GetLfoValue
        let mut value: i64 = if p.depth == 0 || *delay_counter < p.delay {
            0
        } else {
            i64::from(snd_sin_idx((*counter as u32 >> 8) as i32))
                * i64::from(p.depth)
                * i64::from(p.range)
        };
        if target_scale && value != 0 {
            match p.lfo_type {
                lfo_type::VOLUME => value *= 60,
                lfo_type::PITCH | lfo_type::PAN => value <<= 6,
                _ => {}
            }
            value >>= 14;
        }
        // SND_UpdateLfo
        if *delay_counter < p.delay {
            *delay_counter += 1;
        } else {
            let mut tmp = *counter;
            tmp += p.speed << 6;
            tmp >>= 8;
            while tmp >= 0x80 {
                tmp -= 0x80;
            }
            *counter += p.speed << 6;
            *counter &= 0xFF;
            *counter |= tmp << 8;
        }
        value
    }

    #[test]
    fn lfo_tick_matches_pokediamond_reference() {
        // Sweep a representative parameter grid and assert lfo_tick tracks the reference
        // SND_GetLfoValue/SND_UpdateLfo pair tick-for-tick (value, phase, and delay counter).
        for &lfo_type in &[lfo_type::VOLUME, lfo_type::PITCH, lfo_type::PAN] {
            for &depth in &[0, 1, 64, 127] {
                for &delay in &[0, 1, 5] {
                    for &speed in &[1, 16, 64] {
                        let p = LfoParams {
                            depth,
                            delay,
                            lfo_type,
                            speed,
                            range: 1,
                        };
                        let (mut c1, mut d1) = (0i32, 0i32);
                        let (mut c2, mut d2) = (0i32, 0i32);
                        for _ in 0..40 {
                            let got = lfo_tick(&p, &mut c1, &mut d1);
                            let want = pokediamond_lfo(&p, &mut c2, &mut d2, true);
                            assert_eq!(got, want, "value mismatch ({p:?})");
                            assert_eq!(c1, c2, "phase counter mismatch");
                            assert_eq!(d1, d2, "delay counter mismatch");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn delayed_lfo_engages_after_delay() {
        // The bug this fixes: a non-zero LFO delay must suppress modulation for exactly `delay`
        // ticks and then engage (rather than being suppressed forever).
        let p = LfoParams {
            depth: 127,
            delay: 4,
            lfo_type: lfo_type::PITCH,
            speed: 16,
            range: 1,
        };
        let (mut counter, mut delay_counter) = (0i32, 0i32);
        // Advance the phase a bit so a sine value is available once the delay elapses. With the
        // phase frozen during the delay, snd_sin_idx(0) == 0, so we seed a non-zero phase to make
        // the "engages" assertion meaningful: instead, check the counter actually advances only
        // after the delay, and that values are zero throughout the delay window.
        for tick in 0..p.delay {
            let v = lfo_tick(&p, &mut counter, &mut delay_counter);
            assert_eq!(v, 0, "tick {tick}: value must be 0 during the delay window");
            assert_eq!(
                counter, 0,
                "tick {tick}: phase must stay frozen during the delay"
            );
            assert_eq!(
                delay_counter,
                tick + 1,
                "tick {tick}: delay counter must count up"
            );
        }
        // Delay has now elapsed: the phase begins advancing.
        assert_eq!(delay_counter, p.delay);
        lfo_tick(&p, &mut counter, &mut delay_counter);
        assert_ne!(counter, 0, "phase must advance once the delay has elapsed");
        // After enough ticks for the phase to leave the sin(0)=0 point, a non-zero modulation
        // value must appear — i.e. the LFO actually engages rather than staying silent forever.
        let mut saw_nonzero = false;
        for _ in 0..64 {
            if lfo_tick(&p, &mut counter, &mut delay_counter) != 0 {
                saw_nonzero = true;
                break;
            }
        }
        assert!(saw_nonzero, "delayed LFO never produced a non-zero value");
    }

    #[test]
    #[ignore = "diagnostic: prints opcodes used by the golden song"]
    fn scan_golden_opcodes() {
        use std::sync::atomic::Ordering;
        let rom = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../demos/super-mario-64-ds.sdat"),
        )
        .unwrap();
        let sdats = crate::Sdat::load_all(&rom);
        let mut c = Controller::new(32768.0, &sdats[0], 0).unwrap();
        let cfg = SynthConfig::default();
        for _ in 0..(32768 * 8) {
            c.next_sample(&cfg);
        }
        let seen: Vec<String> = (0u16..256)
            .filter(|&op| crate::sequence::OPCODE_SEEN[op as usize].load(Ordering::Relaxed))
            .map(|op| format!("{op:#04X}"))
            .collect();
        println!("opcodes used: {}", seen.join(" "));
    }

    #[test]
    fn zero_depth_lfo_is_always_silent() {
        let p = LfoParams {
            depth: 0,
            delay: 0,
            lfo_type: lfo_type::PITCH,
            speed: 16,
            range: 1,
        };
        let (mut counter, mut delay_counter) = (0i32, 0i32);
        for _ in 0..32 {
            assert_eq!(lfo_tick(&p, &mut counter, &mut delay_counter), 0);
        }
    }
}
