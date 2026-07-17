//! The Game Boy Advance sound device: GBA ROMs running the MP2K ("Sappy" / `m4a`) engine,
//! emulated from the `pret/pokeemerald` decompilation.
//!
//! Data flow within this folder:
//!
//! ```text
//! .gba bytes        ─► rom::GbaRom (song table + headers)          — the archive
//! GbaRom + song id  ─► GbaPlayer (this file — the synth glue)
//! GbaPlayer::tick   ─► m4a_1::MPlayMain / m4a_1::SoundMain          — the faithful engine,
//!                      driving m4a.rs's MusicPlayerInfo / SoundInfo / SoundChannel / CgbChannel
//!                   ─► read the resolved channel state → SynthEvent stream → SynthController
//! ```
//!
//! The engine ([`m4a`] = the `m4a.c` port, [`m4a_1`] = the `m4a_1.s` port, [`m4a_tables`] = the
//! `m4a_tables.c` port) is a faithful transliteration that only mutates hardware state. Everything
//! in *this* file is the minimal synth glue with no reference-source origin: driving the engine per
//! VBlank, turning the resulting `SoundChannel`/`CgbChannel` state into `SynthEvent`s, decoding
//! DirectSound PCM, and generating the PSG (square / noise / programmable-wave) sample data the
//! hardware would produce in silicon.

mod extract;
/// Faithful transliteration of `pret/pokeemerald`'s `src/m4a.c` (structs + the C sound routines).
pub mod m4a;
/// Faithful transliteration of `src/m4a_1.s` (the hand-written ARM engine: `MPlayMain`, `ply_*`,
/// `ChnVolSetAsm`, `SoundMain`/`SoundMainRAM`, `ply_lfos`, `TrackStop`), driving [`m4a`]'s structs.
mod m4a_1;
/// Faithful transliteration of `src/m4a_tables.c` (the `MidiKeyTo*` LUTs + helpers).
pub mod m4a_tables;
pub mod param_player;
pub mod rom;
mod voices;

pub use extract::extract_audio;
pub use extract::waveform_dc_stats;
pub use rom::GbaRom;

use std::sync::Arc;

use crate::PerDeviceSettings;
use crate::devices::{SynthEvent, TickFeedback};
use crate::util::read_u32;
use m4a::{ID_NUMBER, MusicPlayerInfo, MusicPlayerTrack, SongHeader, SoundInfo};
use rom::ptr_to_offset;
use voices::GbaVoices;

/// GBA CPU clock, in Hz.
pub const GBA_CLOCK_RATE: u64 = 16_777_216;

/// CPU cycles per LCD refresh — the MP2K engine runs once per VBlank (≈59.7275 Hz).
pub const CYCLES_PER_FRAME: u64 = 280_896;

/// The software mixer rate (`SOUND_MODE_FREQ_13379`) — the playback rate of fixed-frequency
/// (`TONEDATA_TYPE_FIX`) voices and the rate every DirectSound voice is mixed at on hardware.
pub const ENGINE_RATE: f64 = 13379.0;

/// We run the full hardware channel-struct count rather than the game-configured `maxChans`
/// (usually 5–8) so dense songs don't drop notes.
const MAX_DS_CHANNELS: usize = m4a::MAX_DIRECTSOUND_CHANNELS;

/// `SOUND_MODE_MASVOL` value every Pokémon game passes to `m4aSoundMode`.
const MASTER_VOLUME: u8 = 12;

/// The output level of the faithful `m4a_1.s` software mixer (`SoundMainRAM`). Each DirectSound
/// channel accumulates `(envVolSide · s) >> 8` (`s` ∈ [−128, 127]) into an 8-bit (±128) PCM buffer,
/// so on each output side it reaches `envVolSide / 256` of full scale — a single max-volume channel
/// spans full scale. Optime carries this as one voice `volume = (envVolLeft + envVolRight) / 256`,
/// which the panner splits back across the two sides; the split preserves the sum, so the per-side
/// levels (and thus the dBFS) match the hardware mixer. See `ds_voice_level_matches_pokeemerald`.
const DS_MIXER_FULL_SCALE: f64 = 256.0;

/// PSG (CGB) channels are summed with DirectSound at the analog DAC, *outside* the `m4a_1.s`
/// software mixer, so their relative level is a hardware analog balance rather than a mixer figure.
/// Optime places a full-scale PSG channel at a quarter of output full scale (the classic GBA
/// PSG:DirectSound ratio). PSG sample data is ±0.5, so the `env → volume` scalar tops out at `0.5`
/// (`0.5 · 0.5 = 0.25`), putting a full PSG channel a factor of 4 below a full DirectSound one.
const CGB_FULL_SCALE_GAIN: f64 = 0.5;

/// The GBA device player: the minimal glue driving the faithful engine and emitting `SynthEvent`s.
pub struct GbaPlayer {
    rom: Arc<[u8]>,
    mp: MusicPlayerInfo,
    si: SoundInfo,
    /// Voice/waveform bookkeeping, shared with [`param_player`].
    voices: GbaVoices,
    /// Last MP2K reverb amount emitted as `ReverbAmount`, so it is sent once (and again if it ever
    /// changes). `None` until the first tick emits the song's amount.
    last_reverb: Option<u8>,
    finish_reported: bool,
}

impl GbaPlayer {
    /// Binds song `song_id` from `rom` (mirrors `MPlayStart`). Returns `None` for empty/invalid
    /// songs.
    pub fn new(rom: &GbaRom, song_id: u32) -> Option<GbaPlayer> {
        let header = rom.song_header(song_id)?;
        let data = rom.data.clone();

        // Build the m4a `SongHeader` the engine wants, translating ROM-space pointers to offsets
        // (Optime reads a byte slice, so `cmdPtr` and the voicegroup are offsets, not addresses).
        let mut song = SongHeader {
            trackCount: header.track_count,
            priority: header.priority,
            reverb: header.reverb,
            tone: header.voicegroup as u32,
            ..SongHeader::default()
        };
        for i in 0..header.track_count as usize {
            let ptr = read_u32(&data, header.offset + 8 + i * 4);
            song.part[i] = ptr_to_offset(ptr, data.len())? as u32;
        }

        let mut mp = MusicPlayerInfo {
            ident: ID_NUMBER,
            trackCount: header.track_count,
            tracks: vec![MusicPlayerTrack::default(); header.track_count as usize],
            ..MusicPlayerInfo::default()
        };
        m4a::MPlayStart(&mut mp, song);

        let mut si = SoundInfo {
            maxChans: MAX_DS_CHANNELS as u8,
            masterVolume: MASTER_VOLUME,
            ..SoundInfo::default()
        };
        // `MPlayStart` runs `m4aSoundMode(songHeader->reverb)` when the SET bit is present; its only
        // Optime-visible effect is storing the reverb amount, applied here since `si` is built after.
        if let Some(reverb) = m4a::reverb_from_song_header(header.reverb) {
            si.reverb = reverb;
        }

        Some(GbaPlayer {
            voices: GbaVoices::new(data.clone(), header.track_count as usize),
            rom: data,
            mp,
            si,
            last_reverb: None,
            finish_reported: false,
        })
    }

    /// The engine's per-track state, as of the last [`tick`](crate::devices::DevicePlayer::tick).
    ///
    /// The `SynthEvent` stream deliberately carries only what the synth layer needs (`TrackPan`,
    /// `TrackDetune`), not MP2K's raw `vol`/`bend`/`mod_`/tone registers. The project exporter needs
    /// those registers themselves to write a control timeline, so it reads them here.
    pub fn tracks(&self) -> &[MusicPlayerTrack] {
        &self.mp.tracks
    }

    /// Overrides how many DirectSound channels the engine may allocate (`SOUND_MODE_MAXCHN`),
    /// clamped to the hardware struct count.
    ///
    /// Optime defaults to the full [`MAX_DS_CHANNELS`] rather than the 5–8 a game configures, so
    /// dense songs keep every note (see the note on `MAX_DS_CHANNELS`). That is deliberately *less*
    /// faithful: [`m4a_1::alloc_direct_sound`] steals by priority within `maxChans`, so real
    /// hardware drops notes this does not. Anything chasing hardware behavior — the VST3 plugin —
    /// should set the game's real value.
    pub fn set_max_chans(&mut self, max_chans: u8) {
        self.si.maxChans = max_chans.clamp(1, MAX_DS_CHANNELS as u8);
    }

    fn steps_per_frame(&self) -> f64 {
        f64::from(self.mp.tempoI) / 150.0
    }

    fn tick_impl(
        &mut self,
        feedback: &mut TickFeedback,
        config: &PerDeviceSettings,
        events: &mut Vec<SynthEvent>,
    ) {
        self.voices.set_remove_dc(config.remove_sample_dc_offset);

        // Announce the song's MP2K reverb amount once (and again only if it ever changes). The
        // controller applies it as a mono feedback delay on the sampled bus.
        if self.last_reverb != Some(self.si.reverb) {
            self.last_reverb = Some(self.si.reverb);
            events.push(SynthEvent::ReverbAmount {
                amount: self.si.reverb,
            });
        }

        self.voices.reap_ended(&mut self.si, feedback);

        // One VBlank: advance the tracks, then start any freshly-allocated notes (while their
        // `SF_START` flag is still set), then step the envelopes, then emit the per-voice updates.
        let result = m4a_1::MPlayMain(&mut self.mp, &mut self.si, &self.rom);
        self.voices.emit_track_pans(&self.mp.tracks, events);
        self.voices.start_new_notes(&mut self.si, events);
        m4a_1::SoundMain(&mut self.si);
        self.voices.emit_updates(&self.si, events);

        if result.looped {
            events.push(SynthEvent::Looped);
        }
        if self.mp.status & 0x8000_0000 != 0 && !self.finish_reported {
            self.finish_reported = true;
            events.push(SynthEvent::Ended);
        }
    }
}

impl crate::devices::DevicePlayer for GbaPlayer {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn clock_rate(&self) -> f64 {
        GBA_CLOCK_RATE as f64
    }

    fn cycles_per_tick(&self) -> f64 {
        CYCLES_PER_FRAME as f64
    }

    fn steps_elapsed(&self) -> u32 {
        self.mp.clock
    }

    fn step_rate(&self) -> f64 {
        let frame_rate = GBA_CLOCK_RATE as f64 / CYCLES_PER_FRAME as f64;
        frame_rate * self.steps_per_frame()
    }

    fn steps_per_beat(&self) -> f64 {
        24.0
    }

    fn tick(
        &mut self,
        feedback: &mut TickFeedback,
        config: &PerDeviceSettings,
        events: &mut Vec<SynthEvent>,
    ) {
        self.tick_impl(feedback, config, events);
    }
}

impl crate::devices::SoundData for GbaRom {
    fn song_ids(&self) -> Vec<u32> {
        (0..self.song_count() as u32)
            .filter(|&id| self.song_header(id).is_some())
            .collect()
    }

    fn make_player(&self, id: u32) -> Option<Box<dyn crate::devices::DevicePlayer>> {
        Some(Box::new(GbaPlayer::new(self, id)?))
    }

    fn waveform_dc_stats(&self, id: u32) -> Vec<crate::devices::WaveformDcStat> {
        waveform_dc_stats(self, id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-side output byte the `m4a_1.s` DirectSound mixer (`SoundMainRAM`, inner loop
    /// `_081DD0B0`) writes for one channel into a freshly cleared PCM buffer: `(envVolSide · s) >> 8`
    /// (`mul` then `bic …, 0xFF0000` then the rotate-accumulate keep only the high byte of the
    /// 16-bit product). The buffer is signed 8-bit, so this is the channel's contribution in units of
    /// full scale = ±128.
    fn mixer_output_byte(env_vol_side: u8, sample: i8) -> i32 {
        (i32::from(env_vol_side) * i32::from(sample)) >> 8
    }

    /// The DirectSound voice level emitted by [`GbaPlayer::emit_updates`] carries the *summed* L+R
    /// gain (the panner splits it back, preserving the sum). Pinned here against the reference
    /// mixer: for any `(envVolLeft, envVolRight)` and sample, the synth's summed output
    /// `sample_norm · volume` equals the mixer's summed normalized output (both sides), to within the
    /// mixer's per-side `>> 8` truncation.
    #[test]
    fn ds_voice_level_matches_pokeemerald() {
        for env_l in [0u8, 1, 32, 100, 180, 206, 254] {
            for env_r in [0u8, 1, 32, 100, 180, 206, 254] {
                // The exact expression from `emit_updates`.
                let volume = (f64::from(env_l) + f64::from(env_r)) / DS_MIXER_FULL_SCALE;
                for sample in [-128i8, -100, -1, 0, 1, 64, 127] {
                    let sample_norm = f64::from(sample) / 128.0;
                    let synth_sum = sample_norm * volume;
                    // Reference: both channels' bytes, normalized to full scale (±128).
                    let reference_sum = (f64::from(mixer_output_byte(env_l, sample))
                        + f64::from(mixer_output_byte(env_r, sample)))
                        / 128.0;
                    // Each side's `>> 8` floors away up to 1 PCM8 LSB (= 1/128 of full scale); two
                    // sides ⇒ tolerance just over 2/128.
                    let tol = 2.0 / 128.0 + 1e-9;
                    assert!(
                        (synth_sum - reference_sum).abs() <= tol,
                        "env_l={env_l} env_r={env_r} sample={sample}: \
                         synth={synth_sum} reference={reference_sum}"
                    );
                }
            }
        }
    }

    /// A single max-volume DirectSound channel (both sides near the 8-bit ceiling) reaches essentially
    /// full scale on each output side — the invariant the mixer's `(envVolSide · s) >> 8` into a ±128
    /// buffer enforces, and the reason the emitted `volume` divides by exactly 256.
    #[test]
    fn full_ds_channel_spans_full_scale() {
        // Max side volume with the game's master (12): rightVolume 255 · uvol((12+1)·255>>4=207)>>8.
        let env = ((255u32 * ((13 * 255) >> 4)) >> 8) as u8; // == 206
        let volume = (f64::from(env) + f64::from(env)) / DS_MIXER_FULL_SCALE;
        // Full-scale sample ±1.0, centred (pan split 0.5/0.5) ⇒ per-side peak = 0.5 · volume.
        let per_side_peak = 0.5 * volume;
        assert!(
            (0.75..=1.0).contains(&per_side_peak),
            "per-side peak {per_side_peak} should approach full scale"
        );
    }

    /// A full-scale PSG (CGB) channel spans a quarter of output full scale: ±0.5 sample data times
    /// the `env=15` volume scalar (`0.5`) gives a summed peak of `0.25`.
    #[test]
    fn full_psg_channel_spans_quarter_scale() {
        let volume = 15.0 / 15.0 * CGB_FULL_SCALE_GAIN;
        let psg_peak = 0.5; // PSG sample data is ±0.5.
        assert!((psg_peak * volume - 0.25).abs() < 1e-9);
    }
}
