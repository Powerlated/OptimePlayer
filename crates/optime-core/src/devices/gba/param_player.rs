//! [`ParamPlayer`]: the MP2K engine driven by parameters and DAW notes instead of song bytecode.
//!
//! [`GbaPlayer`](super::GbaPlayer) runs a song: `MPlayMain` walks each track's bytecode, and the
//! commands it executes are what move the engine's registers. A DAW has no bytecode — the notes come
//! from a MIDI clip and the registers come from host-automatable parameters. So this is the same
//! engine and the same [`GbaVoices`] glue with a different driver bolted to the front.
//!
//! Everything below the driver is untouched and stays reference-faithful: [`m4a_1::ply_note_with`]
//! resolves the tone and allocates a channel exactly as the sequencer's note command does,
//! [`m4a_1::lfo_step`] / [`m4a_1::gate_tick`] / [`m4a_1::refresh_changed_tracks`] run per frame in
//! `MPlayMain`'s order, and `SoundMain` steps the envelopes.
//!
//! What this module replaces is only the middle of `MPlayMain`'s `step()`: where that runs
//! `execute_command` until a track owes a wait, this applies [`NoteCommand`]s and [`TrackParams`].

use std::sync::Arc;

use crate::PerDeviceSettings;
use crate::devices::{SynthEvent, TickFeedback};

use super::m4a::{
    self, MPT_FLG_EXIST, MPT_FLG_PITCHG, MPT_FLG_VOLCHG, MusicPlayerInfo, MusicPlayerTrack,
    SoundInfo, TONEDATA_TYPE_RHY, TONEDATA_TYPE_SPL, ToneData,
};
use super::voices::GbaVoices;
use super::{CYCLES_PER_FRAME, GBA_CLOCK_RATE, MAX_DS_CHANNELS, m4a_1};

/// MP2K's tempo accumulator step (`m4a_1`'s `TEMPO_STEP`).
const TEMPO_STEP: u16 = 150;

/// A note the DAW asked for, applied at the start of the next frame.
#[derive(Debug, Clone, Copy)]
pub enum NoteCommand {
    /// Start `key` on `track`. Started with a gate of 0 — see [`ParamPlayer::note_on`].
    On { track: usize, key: u8, velocity: u8 },
    /// Release `key` on `track`.
    Off { track: usize, key: u8 },
}

/// One track's parameter surface: the MP2K registers a host can automate.
///
/// The field order mirrors `m4a_1::control_command`'s own split — the per-track control commands
/// (`0xBA`–`0xC8`) first, then the `XCMD` (`0xCD`) tone overrides. Fields the engine *derives*
/// (`volMR`, `volML`, `keyM`, `pitM`, `modM`, …) are deliberately absent: they are outputs of these,
/// recomputed every frame by `TrkVolPitSet`, so a control for them would do nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackParams {
    /// `0xBD` VOICE — index into the voicegroup.
    pub prog: u8,
    /// `0xBE` VOL.
    pub vol: u8,
    /// `0xBF` PAN, centred (MP2K stores `arg - 0x40`).
    pub pan: i8,
    /// `0xC0` BEND, centred.
    pub bend: i8,
    /// `0xC1` BENDR.
    pub bend_range: u8,
    /// `0xC4` MOD.
    pub mod_: u8,
    /// `0xC2` LFOS.
    pub lfo_speed: u8,
    /// `0xC3` LFODL.
    pub lfo_delay: u8,
    /// `0xC5` MODT — 0 = vibrato (pitch), 1 = tremolo (volume), 2 = auto-pan.
    pub mod_type: u8,
    /// `0xC8` TUNE, centred.
    pub tune: i8,
    /// `0xBC` KEYSH.
    pub key_shift: i8,
    /// `0xBA` PRIO.
    pub priority: u8,

    /// Whether the `XCMD` tone overrides below apply at all.
    ///
    /// Off (the default) means the voicegroup's `ToneData` is used verbatim, which is what a song
    /// does unless it explicitly issues an `XCMD`. This matters beyond taste: without it the
    /// override params would silently fight whatever program `prog` selects, so a fresh plugin
    /// instance would not sound like the ROM.
    pub tone_override: bool,
    /// `ply_xtype` — `tone.kind`.
    pub kind: u8,
    /// `ply_xatta` — `tone.attack`.
    pub attack: u8,
    /// `ply_xdeca` — `tone.decay`.
    pub decay: u8,
    /// `ply_xsust` — `tone.sustain`.
    pub sustain: u8,
    /// `ply_xrele` — `tone.release`.
    pub release: u8,
    /// `ply_xleng` — `tone.length`.
    pub length: u8,
    /// `ply_xswee` — `tone.pan_sweep`.
    pub pan_sweep: u8,
    /// `ply_xiecv` — `pseudoEchoVolume`.
    pub echo_volume: u8,
    /// `ply_xiecl` — `pseudoEchoLength`.
    pub echo_length: u8,
}

impl Default for TrackParams {
    /// MP2K's own track defaults (`MPlayMain`'s `MPT_FLG_START` reset), so an untouched plugin
    /// track starts where a freshly started song track does.
    fn default() -> Self {
        TrackParams {
            prog: 0,
            vol: 0,
            pan: 0,
            bend: 0,
            bend_range: 2,
            mod_: 0,
            lfo_speed: 0x16,
            lfo_delay: 0,
            mod_type: 0,
            tune: 0,
            key_shift: 0,
            priority: 0,
            tone_override: false,
            kind: 1,
            attack: 0xFF,
            decay: 0,
            sustain: 0xFF,
            release: 0,
            length: 0,
            pan_sweep: 0,
            echo_volume: 0,
            echo_length: 0,
        }
    }
}

impl TrackParams {
    /// The parameters a *running song's* track currently holds — the inverse of
    /// [`ParamPlayer::set_track_params`], for rip capture.
    ///
    /// `prog` has to be supplied because MP2K does not keep it: `VOICE` (`0xBD`) copies the
    /// voicegroup's record into the track and throws the index away. [`program_of`] recovers it.
    ///
    /// Every field here is one the engine *stores*; the derived ones (`volMR`, `keyM`, `modM`, …)
    /// are deliberately not read back — they are recomputed from these, so capturing them would
    /// record an output as if it were an input.
    pub fn from_track(track: &MusicPlayerTrack, prog: u8, tone_override: bool) -> TrackParams {
        TrackParams {
            prog,
            vol: track.vol,
            pan: track.pan,
            bend: track.bend,
            bend_range: track.bendRange,
            mod_: track.mod_,
            lfo_speed: track.lfoSpeed,
            lfo_delay: track.lfoDelay,
            mod_type: track.modT,
            tune: track.tune,
            key_shift: track.keyShift,
            priority: track.priority,
            tone_override,
            kind: track.tone.kind,
            attack: track.tone.attack,
            decay: track.tone.decay,
            sustain: track.tone.sustain,
            release: track.tone.release,
            length: track.tone.length,
            pan_sweep: track.tone.pan_sweep,
            echo_volume: track.pseudoEchoVolume,
            echo_length: track.pseudoEchoLength,
        }
    }
}

/// Recovers which program index in `voicegroup` a track's `tone` came from.
///
/// `VOICE` (`0xBD`) copies a `ToneData` out of the voicegroup and keeps no index, so the only way
/// back is to search. Returns `None` when no record matches — which is itself the useful signal:
/// the song must have edited the tone with an `XCMD`, so a capture should record the tone fields
/// as an override rather than a program change.
///
/// The scan is 128 twelve-byte reads; callers cache on the tone, which changes rarely.
pub fn program_of(rom: &[u8], voicegroup: usize, tone: &ToneData) -> Option<u8> {
    (0..128u8).find(|&i| ToneData::read(rom, voicegroup + i as usize * 12) == *tone)
}

/// The MP2K engine with a parameter/note driver in front of it.
pub struct ParamPlayer {
    rom: Arc<[u8]>,
    mp: MusicPlayerInfo,
    si: SoundInfo,
    voices: GbaVoices,
    /// ROM offset of the voicegroup `prog` indexes into.
    voicegroup: usize,
    /// Per-track parameters as last applied, so a frame only touches what changed.
    params: Vec<TrackParams>,
    pending: Vec<NoteCommand>,
    last_reverb: Option<u8>,
}

impl ParamPlayer {
    /// Builds a player over `rom`, taking programs from the voicegroup at ROM offset `voicegroup`.
    ///
    /// `track_count` is how many tracks the driver exposes (the plugin uses
    /// [`MAX_MUSICPLAYER_TRACKS`](super::m4a::MAX_MUSICPLAYER_TRACKS) = 16, one per MIDI channel).
    pub fn new(rom: Arc<[u8]>, voicegroup: usize, track_count: usize) -> ParamPlayer {
        // Mirror `MPlayMain`'s `MPT_FLG_START` reset: every track exists (so the per-frame passes
        // don't skip it), with MP2K's defaults.
        let mut tracks = vec![MusicPlayerTrack::default(); track_count];
        for tr in &mut tracks {
            tr.flags = MPT_FLG_EXIST;
            tr.bendRange = 2;
            tr.volX = 0x40;
            tr.lfoSpeed = 0x16;
            tr.tone.kind = 1;
        }

        let mp = MusicPlayerInfo {
            ident: m4a::ID_NUMBER,
            trackCount: track_count as u8,
            tracks,
            tempoI: 150,
            ..MusicPlayerInfo::default()
        };
        let si = SoundInfo {
            maxChans: MAX_DS_CHANNELS as u8,
            masterVolume: 12,
            ..SoundInfo::default()
        };

        let mut player = ParamPlayer {
            voices: GbaVoices::new(rom.clone(), track_count),
            rom,
            mp,
            si,
            voicegroup,
            params: vec![TrackParams::default(); track_count],
            pending: Vec::new(),
            last_reverb: None,
        };
        // Resolve each track's starting tone. `set_track_params` only re-reads the voicegroup when
        // `prog` *changes*, and a fresh track already sits at the default program — so without this
        // the tone would keep `MusicPlayerTrack::default()`'s null `wav` and every note would be
        // silently dropped for having no waveform.
        for t in 0..track_count {
            player.load_tone(t, &TrackParams::default());
        }
        player
    }

    /// The engine's per-track state, as of the last tick. See [`TrackParams::from_track`].
    pub fn tracks(&self) -> &[MusicPlayerTrack] {
        &self.mp.tracks
    }

    /// Queues a note-on, applied at the start of the next frame.
    ///
    /// **MP2K has no note-off.** A song bakes a gate time into the note command and `gate_tick`
    /// releases the channel when it expires; a DAW's note-off is asynchronous and unknown at
    /// note-on. So notes start with a gate of 0, which is MP2K's own tie (`0xCE`): `gate_tick`
    /// never counts a zero gate down, so the note holds until [`Self::note_off`] releases it.
    pub fn note_on(&mut self, track: usize, key: u8, velocity: u8) {
        self.pending.push(NoteCommand::On {
            track,
            key,
            velocity,
        });
    }

    /// Queues a note-off, applied at the start of the next frame.
    pub fn note_off(&mut self, track: usize, key: u8) {
        self.pending.push(NoteCommand::Off { track, key });
    }

    /// Sets the tempo the LFO and gate passes run at. MP2K's `tempoI` is ~BPM (`steps_per_beat` is
    /// 24 and a step is `tempoI / 150` of a VBlank), so the host's tempo maps straight onto it.
    pub fn set_tempo_bpm(&mut self, bpm: f64) {
        self.mp.tempoI = bpm.clamp(1.0, 511.0).round() as u16;
    }

    /// `SOUND_MODE_MASVOL` (0..=15).
    pub fn set_master_volume(&mut self, volume: u8) {
        self.si.masterVolume = volume.min(15);
    }

    /// `SOUND_MODE_REVERB` (0..=127).
    pub fn set_reverb(&mut self, reverb: u8) {
        self.si.reverb = reverb.min(127);
    }

    /// `SOUND_MODE_MAXCHN` — how many DirectSound channels may be allocated.
    ///
    /// Games configure 5–8 and `alloc_direct_sound` steals by priority within it, so this is a
    /// fidelity control, not a performance one: a higher value keeps notes hardware would drop.
    pub fn set_max_chans(&mut self, max_chans: u8) {
        self.si.maxChans = max_chans.clamp(1, MAX_DS_CHANNELS as u8);
    }

    /// Points `prog` at a different voicegroup (a ROM offset). Takes effect on the next note.
    pub fn set_voicegroup(&mut self, voicegroup: usize) {
        self.voicegroup = voicegroup;
        // Re-resolve every track's tone against the new group.
        for t in 0..self.params.len() {
            let params = self.params[t];
            self.load_tone(t, &params);
        }
    }

    /// Applies `params` to track `t`, raising the same flags the corresponding MP2K commands do.
    ///
    /// The flags are the point: `refresh_changed_tracks` only recomputes a track's mixers and its
    /// channels' pitch when `MPT_FLG_VOLCHG`/`PITCHG` is set, so a register written without its flag
    /// silently does nothing until the next note lands.
    pub fn set_track_params(&mut self, t: usize, params: &TrackParams) {
        let Some(old) = self.params.get(t).copied() else {
            return;
        };
        if old == *params {
            return;
        }

        // The tone is re-resolved whenever the program or any override changes; `load_tone` reads
        // the voicegroup and then applies the overrides on top.
        if old.prog != params.prog
            || old.tone_override != params.tone_override
            || (params.tone_override && tone_fields_differ(&old, params))
        {
            self.load_tone(t, params);
        }

        let tr = &mut self.mp.tracks[t];

        // `0xBE` VOL / `0xBF` PAN → VOLCHG.
        if old.vol != params.vol {
            tr.vol = params.vol;
            tr.flags |= MPT_FLG_VOLCHG;
        }
        if old.pan != params.pan {
            tr.pan = params.pan;
            tr.flags |= MPT_FLG_VOLCHG;
        }

        // `0xBC` KEYSH / `0xC0` BEND / `0xC1` BENDR / `0xC8` TUNE → PITCHG.
        if old.key_shift != params.key_shift {
            tr.keyShift = params.key_shift;
            tr.flags |= MPT_FLG_PITCHG;
        }
        if old.bend != params.bend {
            tr.bend = params.bend;
            tr.flags |= MPT_FLG_PITCHG;
        }
        if old.bend_range != params.bend_range {
            tr.bendRange = params.bend_range;
            tr.flags |= MPT_FLG_PITCHG;
        }
        if old.tune != params.tune {
            tr.tune = params.tune;
            tr.flags |= MPT_FLG_PITCHG;
        }

        // `0xC2` LFOS / `0xC4` MOD — both clear the accumulated depth when set to 0, exactly as
        // their command handlers do; otherwise a stale `modM` keeps modulating a switched-off LFO.
        if old.lfo_speed != params.lfo_speed {
            tr.lfoSpeed = params.lfo_speed;
            if tr.lfoSpeed == 0 {
                m4a::ClearModM(tr);
            }
        }
        if old.mod_ != params.mod_ {
            tr.mod_ = params.mod_;
            if tr.mod_ == 0 {
                m4a::ClearModM(tr);
            }
        }
        if old.lfo_delay != params.lfo_delay {
            tr.lfoDelay = params.lfo_delay;
        }

        // `0xC5` MODT retargets the LFO, so both mixers are stale.
        if old.mod_type != params.mod_type {
            tr.modT = params.mod_type;
            tr.flags |= MPT_FLG_VOLCHG | MPT_FLG_PITCHG;
        }

        // `0xBA` PRIO.
        if old.priority != params.priority {
            tr.priority = params.priority;
        }

        // `ply_xiecv` / `ply_xiecl` live on the track, not the tone.
        tr.pseudoEchoVolume = params.echo_volume;
        tr.pseudoEchoLength = params.echo_length;

        self.params[t] = *params;
    }

    /// Resolves track `t`'s tone: the voicegroup's record for `prog`, plus the `XCMD` overrides.
    fn load_tone(&mut self, t: usize, params: &TrackParams) {
        // `0xBD` VOICE: copy the program's ToneData from the voicegroup.
        let mut tone = ToneData::read(&self.rom, self.voicegroup + params.prog as usize * 12);

        if params.tone_override {
            // A key-split / rhythm record stores a *pointer* in its attack/decay/sustain/release
            // bytes (`ply_note_with` reassembles them into the split table address), not envelope
            // rates. Overriding them would corrupt that pointer and the note would resolve to
            // garbage, so the envelope overrides only apply to a plain tone. The rest are safe.
            let is_split = tone.kind & (TONEDATA_TYPE_RHY | TONEDATA_TYPE_SPL) != 0;
            if !is_split {
                tone.attack = params.attack;
                tone.decay = params.decay;
                tone.sustain = params.sustain;
                tone.release = params.release;
            }
            tone.kind = params.kind;
            tone.length = params.length;
            tone.pan_sweep = params.pan_sweep;
        }

        self.mp.tracks[t].tone = tone;
    }

    /// Releases every note sounding on `track`, plus any note-on still queued for it.
    ///
    /// For a transport stop or jump: a DAW will not send note-offs for notes it is abandoning, so
    /// without this a held note (gate 0 — see [`Self::note_on`]) would ring forever. Uses the
    /// engine's release rather than cutting the channel, so a stop sounds like a key lift.
    pub fn all_notes_off(&mut self, track: usize) {
        self.pending.retain(|cmd| match cmd {
            NoteCommand::On { track: t, .. } => *t != track,
            NoteCommand::Off { .. } => true,
        });
        for i in 0..MAX_DS_CHANNELS {
            let c = &mut self.si.chans[i];
            if c.track == Some(track) {
                c.statusFlags |= m4a::SOUND_CHANNEL_SF_STOP;
            }
        }
        for i in 0..4 {
            let c = &mut self.si.cgbChans[i];
            if c.track == Some(track) {
                c.statusFlags |= m4a::SOUND_CHANNEL_SF_STOP;
            }
        }
    }

    /// Releases every channel on `track` sounding `key`, the way an expired gate does.
    ///
    /// `gate_tick` ends a note by setting `SF_STOP` and letting the envelope run its release; doing
    /// the same here means a DAW note-off gets the engine's real release rather than a cut.
    fn release(&mut self, track: usize, key: u8) {
        for i in 0..MAX_DS_CHANNELS {
            let c = &mut self.si.chans[i];
            if c.track == Some(track) && c.midiKey == key {
                c.statusFlags |= m4a::SOUND_CHANNEL_SF_STOP;
            }
        }
        for i in 0..4 {
            let c = &mut self.si.cgbChans[i];
            if c.track == Some(track) && c.midiKey == key {
                c.statusFlags |= m4a::SOUND_CHANNEL_SF_STOP;
            }
        }
    }

    fn steps_per_frame(&self) -> f64 {
        f64::from(self.mp.tempoI) / f64::from(TEMPO_STEP)
    }

    /// One VBlank, in `MPlayMain`'s order — with the bytecode walk replaced by the queued notes.
    fn tick_impl(
        &mut self,
        feedback: &mut TickFeedback,
        config: &PerDeviceSettings,
        events: &mut Vec<SynthEvent>,
    ) {
        self.voices.set_remove_dc(config.remove_sample_dc_offset);

        if self.last_reverb != Some(self.si.reverb) {
            self.last_reverb = Some(self.si.reverb);
            events.push(SynthEvent::ReverbAmount {
                amount: self.si.reverb,
            });
        }

        self.voices.reap_ended(&mut self.si, feedback);

        // The driver: what `step()` would have got from `execute_command`.
        for cmd in std::mem::take(&mut self.pending) {
            match cmd {
                NoteCommand::On {
                    track,
                    key,
                    velocity,
                } if track < self.mp.tracks.len() => {
                    // Gate 0 == MP2K's tie: hold until `note_off`.
                    m4a_1::ply_note_with(
                        &mut self.mp,
                        &mut self.si,
                        &self.rom,
                        track,
                        key,
                        velocity,
                        0,
                    );
                }
                NoteCommand::Off { track, key } if track < self.mp.tracks.len() => {
                    self.release(track, key);
                }
                _ => {}
            }
        }

        // The rest of `MPlayMain`: tempo-paced gate/LFO steps, then the changed-track refresh.
        self.mp.tempoC += self.mp.tempoI;
        while self.mp.tempoC >= TEMPO_STEP {
            self.mp.tempoC -= TEMPO_STEP;
            for t in 0..self.mp.tracks.len() {
                m4a_1::gate_tick(&mut self.si, t);
                m4a_1::lfo_step(&mut self.mp.tracks[t]);
            }
            self.mp.clock += 1;
        }
        m4a_1::refresh_changed_tracks(&mut self.mp, &mut self.si, &self.rom);

        self.voices.emit_track_pans(&self.mp.tracks, events);
        self.voices.start_new_notes(&mut self.si, events);
        m4a_1::SoundMain(&mut self.si);
        self.voices.emit_updates(&self.si, events);
    }
}

impl crate::devices::DevicePlayer for ParamPlayer {
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

/// Whether any `XCMD`-backed tone field differs between two parameter sets.
fn tone_fields_differ(a: &TrackParams, b: &TrackParams) -> bool {
    a.kind != b.kind
        || a.attack != b.attack
        || a.decay != b.decay
        || a.sustain != b.sustain
        || a.release != b.release
        || a.length != b.length
        || a.pan_sweep != b.pan_sweep
}
