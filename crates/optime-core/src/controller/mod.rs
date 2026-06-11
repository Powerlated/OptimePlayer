//! The audio-side runtime: note lifecycle, ADSR/LFO processing, the DS master clock, and the
//! parallel look-ahead [`FsVisController`] used to drive visualizers.
//!
//! The pieces are split across this module's children:
//! - [`config`] — the [`SynthConfig`] options struct.
//! - [`volume`] / [`lfo`] — the decibel-domain channel-volume and LFO math (pokediamond ports).
//! - [`envelope`] — the per-tick ADSR/LFO pass over the active notes.
//! - [`messages`] — applying sequence messages and starting notes (incl. live keyboard input).
//! - [`vis`] — the look-ahead [`FsVisController`].

mod config;
mod envelope;
mod lfo;
mod messages;
mod vis;
mod volume;

pub use config::SynthConfig;
pub use vis::{FsVisController, PitchBendEvent};
pub use volume::calc_channel_volume;

use std::sync::Arc;

use crate::sample::{decode_adpcm, decode_pcm16, decode_pcm8, Sample};
use crate::sdat::Sdat;
use crate::sequence::Sequence;
use crate::synth::MAX_BLOCK;
use crate::tables::SQUARE_WAVES;
use crate::util::{read_u16, read_u32, read_u8};
use crate::{SampleSynthesizer, CYCLES_PER_TICK, DS_CLOCK_RATE, TRACK_COUNT};

/// ADSR envelope stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdsrState {
    Attack,
    Decay,
    Sustain,
    Release,
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
    /// Volume-LFO contribution (dB) computed this tick, summed into the channel volume.
    lfo_vol_db: i32,
    // Resolved instrument coefficients for this note's region.
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
    instrument_bank: crate::bank::InstrumentBank,
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
    /// Renders in blocks between sequencer ticks (voice parameters only change on ticks), so each
    /// voice runs one tight loop per block instead of re-deriving its setup per sample. The clock
    /// is advanced with the same per-sample additions as [`Self::next_sample`], so the output is
    /// bit-identical to calling that in a loop.
    pub fn fill(&mut self, out: &mut [f32], config: &SynthConfig) {
        let threshold = CYCLES_PER_TICK as f64 * self.sample_rate;
        let clock = DS_CLOCK_RATE as f64;
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
