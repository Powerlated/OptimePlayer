//! [`SynthController`]: the device-agnostic synthesis runtime.
//!
//! It owns the per-track voice pools, the master clock, and the note-grid bookkeeping for
//! visualizers. All musical intelligence lives in the device player
//! ([`DevicePlayer`](crate::devices::DevicePlayer)); the controller drives it one tick at a
//! time and applies the standardized [`SynthEvent`] stream it produces to the voices.
//!
//! - [`config`] — the [`SynthConfig`] options struct.
//! - [`vis`] — the look-ahead [`FsVisController`] for visualizers.

mod config;
mod vis;

pub use config::SynthConfig;
pub use vis::{FsVisController, VisNote};

use crate::devices::{DevicePlayer, SoundData, SynthEvent, TickFeedback, VoiceId};
use crate::synth::MAX_BLOCK;
use crate::{SampleSynthesizer, TRACK_COUNT};

/// Which device voice currently owns a pool slot.
#[derive(Debug, Clone, Copy)]
struct SlotOwner {
    voice: VoiceId,
    key: u8,
}

/// The synthesis runtime: voice pools + master clock, driven by a device player.
pub struct SynthController {
    sample_rate: f64,
    /// The device player generating the event stream.
    pub player: DevicePlayer,
    /// One polyphonic synthesizer per track.
    pub synthesizers: Vec<SampleSynthesizer>,
    /// `slot_owner[track][pool_slot]` — which device voice occupies each synthesizer voice.
    slot_owner: Vec<Vec<Option<SlotOwner>>>,
    /// `notes_on[track][note]` is 1 while a sequence note sounds (drives the visualizer).
    pub notes_on: Vec<[u8; 128]>,
    /// As [`Self::notes_on`] but for live keyboard input.
    pub notes_on_keyboard: Vec<[u8; 128]>,
    /// Count of sequence loops seen (used by callers to detect loop points).
    pub jumps: u32,
    /// Set when the song has ended and should fade out.
    pub fading_start: bool,
    /// Synth-side voice endings reported back to the device on the next tick.
    feedback: TickFeedback,
    /// Reusable event buffer.
    events: Vec<SynthEvent>,
    timer: f64,
}

impl SynthController {
    /// Binds song `song_id` from `data` for playback at `sample_rate`.
    ///
    /// Returns `None` if the song is missing or malformed.
    pub fn new(sample_rate: f64, data: &SoundData, song_id: u32) -> Option<SynthController> {
        let player = data.make_player(song_id)?;
        let chain = player.hardware_chain();
        let synthesizers: Vec<_> = (0..TRACK_COUNT)
            .map(|_| SampleSynthesizer::new(sample_rate, 16, chain))
            .collect();
        let slot_owner = synthesizers
            .iter()
            .map(|s| vec![None; s.voice_count()])
            .collect();
        Some(SynthController {
            sample_rate,
            player,
            synthesizers,
            slot_owner,
            notes_on: vec![[0u8; 128]; TRACK_COUNT],
            notes_on_keyboard: vec![[0u8; 128]; TRACK_COUNT],
            jumps: 0,
            fading_start: false,
            feedback: TickFeedback::default(),
            events: Vec::new(),
            timer: 0.0,
        })
    }

    /// The audio sample rate this controller renders at.
    #[inline]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Sequencer steps executed (the visualizer timeline position).
    pub fn steps_elapsed(&self) -> u32 {
        self.player.steps_elapsed()
    }

    /// Current sequencer step rate (steps per second at the current tempo).
    pub fn step_rate(&self) -> f64 {
        self.player.step_rate()
    }

    /// Advances the device master clock and returns one mixed stereo sample.
    ///
    /// This is the single place where the hardware tick math lives: the device clock is
    /// accumulated per output sample and the player is ticked every `cycles_per_tick` cycles.
    pub fn next_sample(&mut self, config: &SynthConfig) -> (f32, f32) {
        self.timer += self.player.clock_rate();
        let threshold = self.player.cycles_per_tick() * self.sample_rate;
        while self.timer >= threshold {
            self.timer -= threshold;
            self.tick(config);
        }

        let mut val_l = 0.0;
        let mut val_r = 0.0;
        for (synth, &enabled) in self.synthesizers.iter_mut().zip(&config.track_enables) {
            synth.next_sample(config);
            if enabled {
                val_l += synth.val_l;
                val_r += synth.val_r;
            }
        }
        (val_l as f32, val_r as f32)
    }

    /// Fills `out` with interleaved stereo (L, R, L, R, …) samples.
    ///
    /// Renders in blocks between device ticks (voice parameters only change on ticks), so each
    /// voice runs one tight loop per block instead of re-deriving its setup per sample. The clock
    /// is advanced with the same per-sample additions as [`Self::next_sample`], so the output is
    /// bit-identical to calling that in a loop.
    pub fn fill(&mut self, out: &mut [f32], config: &SynthConfig) {
        let threshold = self.player.cycles_per_tick() * self.sample_rate;
        let clock = self.player.clock_rate();
        let frames = out.len() / 2;
        let mut acc_l = [0.0f64; MAX_BLOCK];
        let mut acc_r = [0.0f64; MAX_BLOCK];

        let mut frame = 0;
        while frame < frames {
            // First sample of the block: advance the clock and run any due ticks (mirroring
            // `next_sample`'s ordering: the tick fires before the sample is synthesized).
            self.timer += clock;
            while self.timer >= threshold {
                self.timer -= threshold;
                self.tick(config);
            }
            // Extend the block with tick-free samples, advancing the clock identically.
            let max_n = (frames - frame).min(MAX_BLOCK);
            let mut n = 1;
            while n < max_n && self.timer + clock < threshold {
                self.timer += clock;
                n += 1;
            }

            acc_l[..n].fill(0.0);
            acc_r[..n].fill(0.0);
            for (synth, &enabled) in self.synthesizers.iter_mut().zip(&config.track_enables) {
                synth.render_block(config, n, &mut acc_l, &mut acc_r, enabled);
            }
            let block_out = &mut out[2 * frame..2 * (frame + n)];
            for (frame_out, (&l, &r)) in block_out
                .chunks_exact_mut(2)
                .zip(acc_l[..n].iter().zip(&acc_r[..n]))
            {
                frame_out[0] = l as f32;
                frame_out[1] = r as f32;
            }
            frame += n;
        }

        // Odd trailing f32 (half a frame): render one more stereo sample, keep its left channel.
        if out.len() % 2 == 1 {
            let (l, _) = self.next_sample(config);
            out[out.len() - 1] = l;
        }
    }

    /// One device tick: report synth-side voice endings, advance the device, apply its events.
    pub fn tick(&mut self, config: &SynthConfig) {
        // One-shot samples that ran out are cut here — only the synthesizer knows their playback
        // position — and reported to the device as feedback.
        for (t, synth) in self.synthesizers.iter_mut().enumerate() {
            for slot in 0..self.slot_owner[t].len() {
                let Some(owner) = self.slot_owner[t][slot] else {
                    continue;
                };
                let instr = synth.instr(slot);
                let ran_out =
                    !instr.sample.looping && instr.sample_t > instr.sample.data.len() as f64;
                if ran_out || !instr.playing {
                    synth.cut_instrument(slot);
                    self.slot_owner[t][slot] = None;
                    self.notes_on[t][owner.key as usize] = 0;
                    self.feedback.ended_voices.push((t, owner.voice));
                }
            }
        }

        let mut events = std::mem::take(&mut self.events);
        self.player.tick(&mut self.feedback, config, &mut events);
        for event in events.drain(..) {
            self.apply_event(event, config);
        }
        self.events = events;
    }

    /// Applies one standardized device event to the voice pools.
    fn apply_event(&mut self, event: SynthEvent, config: &SynthConfig) {
        match event {
            SynthEvent::NoteStarted {
                track,
                voice,
                key,
                keyboard,
                sample,
                pitch,
                volume,
                duration_ticks: _,
            } => {
                let slot = self.synthesizers[track].play(sample, pitch, volume, config);
                if let Some(old) = self.slot_owner[track][slot].replace(SlotOwner { voice, key }) {
                    // Round-robin steal: the previous occupant is gone; tell the device.
                    self.notes_on[track][old.key as usize] = 0;
                    self.feedback.ended_voices.push((track, old.voice));
                }
                self.notes_on[track][key as usize] = 1;
                if keyboard {
                    self.notes_on_keyboard[track][key as usize] = 1;
                }
            }
            SynthEvent::VoiceVolume {
                track,
                voice,
                volume,
            } => {
                if let Some(slot) = self.find_slot(track, voice) {
                    self.synthesizers[track].instr_mut(slot).volume = volume;
                }
            }
            SynthEvent::VoicePitch {
                track,
                voice,
                pitch,
            } => {
                if let Some(slot) = self.find_slot(track, voice) {
                    self.synthesizers[track]
                        .instr_mut(slot)
                        .set_pitch(pitch, config.tuning);
                }
            }
            SynthEvent::VoiceDetune {
                track,
                voice,
                semitones,
            } => {
                if let Some(slot) = self.find_slot(track, voice) {
                    self.synthesizers[track]
                        .instr_mut(slot)
                        .set_finetune_lfo(semitones, config.tuning);
                }
            }
            SynthEvent::VoiceStopped { track, voice } => {
                if let Some(slot) = self.find_slot(track, voice) {
                    let owner = self.slot_owner[track][slot].take();
                    self.synthesizers[track].stop_instrument(slot, config.smooth_psg_pops);
                    if let Some(owner) = owner {
                        self.notes_on[track][owner.key as usize] = 0;
                    }
                }
            }
            SynthEvent::NoteReleased {
                track,
                key,
                keyboard,
            } => {
                if keyboard {
                    self.notes_on_keyboard[track][key as usize] = 0;
                } else {
                    self.notes_on[track][key as usize] = 0;
                }
            }
            SynthEvent::TrackPan { track, pan } => {
                self.synthesizers[track].set_pan(pan, config);
            }
            SynthEvent::TrackDetune { track, semitones } => {
                self.synthesizers[track].set_finetune(semitones, config.tuning);
            }
            SynthEvent::Looped => self.jumps += 1,
            SynthEvent::Ended => self.fading_start = true,
        }
    }

    /// Finds the pool slot owned by `voice` on `track`.
    fn find_slot(&self, track: usize, voice: VoiceId) -> Option<usize> {
        self.slot_owner[track]
            .iter()
            .position(|o| o.is_some_and(|o| o.voice == voice))
    }

    /// Which track receives live keyboard input, if any.
    pub fn keyboard_track(&self) -> Option<usize> {
        self.player.keyboard_track()
    }

    /// Routes live keyboard input to `track` (or back to the sequence with `None`).
    pub fn set_keyboard_track(&mut self, track: Option<usize>) {
        self.player.set_keyboard_track(track);
    }

    /// Triggers a live keyboard note. The note sounds until released.
    pub fn play_keyboard_note(
        &mut self,
        track: usize,
        note: u8,
        velocity: i32,
        duration: u32,
        config: &SynthConfig,
    ) {
        let mut events = std::mem::take(&mut self.events);
        self.player
            .keyboard_note_on(track, note, velocity, duration, config, &mut events);
        for event in events.drain(..) {
            self.apply_event(event, config);
        }
        self.events = events;
    }

    /// Releases a previously-triggered keyboard note (it enters its release stage).
    pub fn release_keyboard_note(&mut self, track: usize, note: u8, config: &SynthConfig) {
        let mut events = std::mem::take(&mut self.events);
        self.player.keyboard_note_off(track, note, &mut events);
        for event in events.drain(..) {
            self.apply_event(event, config);
        }
        self.events = events;
    }
}
