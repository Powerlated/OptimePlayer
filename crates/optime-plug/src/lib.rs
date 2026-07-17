//! Optime Player as a VST3 instrument: the faithful GBA MP2K engine, driven by a DAW.
//!
//! [`optime_core`] already emulates MP2K accurately enough to be pinned against `pret/pokeemerald`
//! by transcription tests. What it has never had is a way in from outside a ROM: a song's bytecode
//! drives everything. This plugin swaps that driver — notes come from the host's clip and the
//! engine's registers come from VST3 parameters — while everything below the driver stays the same
//! emulation the app plays songs with (see `optime_core::devices::gba::param_player`).
//!
//! The intended workflow is a High Quality Video Game Rip: export a song's notes to MIDI, import it
//! here, and edit against an engine that sounds like the console rather than like a sampler.

pub mod editor;
pub mod params;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use nice_plug::prelude::*;
use optime_core::devices::gba::param_player::ParamPlayer;
use optime_core::{PerDeviceSettings, SynthController};

use editor::{CaptureState, RomSlot};
use optime_core::devices::DevicePlayer;
use optime_core::devices::gba::GbaPlayer;
use optime_core::devices::gba::m4a::ToneData;
use optime_core::devices::gba::param_player::{TrackParams as EngineTrackParams, program_of};
use params::{OptimePlugParams, Preset, TRACKS};

/// Longest run rendered between note events. Blocks are split on events anyway, so this only caps
/// how long a parameter change waits to be applied.
const MAX_BLOCK: usize = 64;

/// Work the editor hands off to a background thread.
///
/// Both of these touch the filesystem and build an engine, so neither may run on the audio thread —
/// and the ROM picker additionally must not run on the *GUI* thread, since a modal file dialog
/// pumps a message loop that would re-enter egui's in-progress draw and crash the host.
#[derive(Debug, Clone, Copy)]
pub enum Task {
    /// Show the file dialog, then open whatever was picked.
    PickRom,
    /// Point the engine at the newly selected song (voicegroup comes from the song header).
    SelectSong,
    /// Build the song engine (the real bytecode player) and arm: the song won't start until the
    /// transport rolls from bar 1 — see [`OptimePlug::process`].
    ArmCapture,
    /// Disarm (armed but not yet started) or stop (running): clears both flags and restores the
    /// note-driven player, but only if capture had actually begun.
    StopCapture,
}

pub struct OptimePlug {
    params: Arc<OptimePlugParams>,
    /// The engine. `None` until a ROM is loaded — a fresh instance has no samples and stays silent.
    controller: Option<SynthController>,
    /// Interleaved scratch: the core renders stereo-interleaved, a `Buffer` is planar.
    scratch: Vec<f32>,
    /// The settings the current `preset` resolves to, refreshed when it changes.
    settings: PerDeviceSettings,
    preset: Preset,
    sample_rate: f64,
    /// Where the background worker parks a freshly built engine for us to swap in.
    rom: Arc<RomSlot>,
    /// What rip capture publishes for the editor to record.
    capture: Arc<CaptureState>,
    /// `program_of` cache, per track: the tone last seen and the program it resolved to, so the
    /// 128-entry voicegroup scan only runs when a track's tone actually changes.
    prog_cache: Vec<Option<(ToneData, Option<u8>)>>,
}

impl Default for OptimePlug {
    fn default() -> Self {
        OptimePlug {
            params: Arc::new(OptimePlugParams::default()),
            controller: None,
            scratch: vec![0.0; MAX_BLOCK * 2],
            settings: Preset::Enhanced.settings(),
            preset: Preset::Enhanced,
            sample_rate: 44_100.0,
            rom: Arc::new(RomSlot::default()),
            capture: Arc::new(CaptureState::default()),
            prog_cache: vec![None; TRACKS],
        }
    }
}

impl OptimePlug {
    /// Reopens the persisted ROM (project load).
    ///
    /// Only called from `initialize`, off the audio thread, where the file I/O is allowed. A
    /// failure leaves the plugin silent rather than pretending: there are no samples to play.
    fn load_rom(&mut self) {
        self.controller = None;
        let path = self.params.rom_path.read().unwrap().clone();
        if !path.is_empty() {
            self.rom.open(&path, &self.params);
            self.take_pending_rom();
        }
    }

    /// Reads the running song's registers and publishes them for the editor to record.
    ///
    /// Only does anything while capture is on, when the controller holds a [`GbaPlayer`] — the real
    /// bytecode player. The note-driven player has no song to read.
    fn publish_capture(&mut self) {
        if !self.capture.active.load(Ordering::Relaxed) {
            return;
        }
        let voicegroup = *self.params.voicegroup.read().unwrap() as usize;
        let rom = match self.rom.rom.try_lock() {
            Ok(rom) => match rom.as_ref() {
                Some(rom) => rom.data.clone(),
                None => return,
            },
            Err(_) => return,
        };

        let Some(controller) = self.controller.as_mut() else {
            return;
        };
        let Some(player) = controller
            .player_mut()
            .as_any_mut()
            .downcast_mut::<GbaPlayer>()
        else {
            return;
        };

        // `try_lock`: the editor may be reading. Skipping is harmless — the next block republishes.
        let Ok(mut out) = self.capture.tracks.try_lock() else {
            return;
        };
        out.clear();
        for (t, track) in player.tracks().iter().enumerate().take(TRACKS) {
            // MP2K discards the VOICE index, so recover it by matching the tone against the
            // voicegroup — cached, since the scan is 128 reads and a tone rarely changes.
            let found = match self.prog_cache[t] {
                Some((tone, prog)) if tone == track.tone => prog,
                _ => {
                    let prog = program_of(&rom, voicegroup, &track.tone);
                    self.prog_cache[t] = Some((track.tone, prog));
                    prog
                }
            };
            // No match means an XCMD edited the tone away from its voicegroup record, so capture it
            // as an override rather than as a program change.
            let (prog, tone_override) = match found {
                Some(prog) => (prog, false),
                None => (0, true),
            };
            out.push(EngineTrackParams::from_track(track, prog, tone_override));
        }
        // The song's own tempo, so the editor can tell the user to match the project to it.
        // `step_rate` is tempo-dependent and `steps_per_beat` is 24 on GBA.
        let bpm = player.step_rate() * 60.0 / player.steps_per_beat();
        drop(out);
        self.capture
            .song_bpm
            .store((bpm as f32).to_bits(), Ordering::Relaxed);
        self.capture.generation.fetch_add(1, Ordering::Release);
    }

    /// Sends the notes the running song produced during the block just rendered.
    ///
    /// The counterpart to the automation snapshot: automation goes out through `performEdit` on the
    /// GUI thread, but *notes* are ordinary events, so they can go straight out of `process` with
    /// sample-accurate timing.
    ///
    /// `offset` is where the rendered block sits in the host's buffer. The controller reports a tap
    /// relative to the block *it* rendered, and `process` renders in `MAX_BLOCK` chunks, so without
    /// the offset every note in a buffer would collapse onto the first 64 samples.
    fn emit_captured_notes(&mut self, context: &mut impl ProcessContext<Self>, offset: u32) {
        if !self.capture.active.load(Ordering::Relaxed) {
            return;
        }
        let Some(controller) = self.controller.as_mut() else {
            return;
        };
        for tap in controller.take_note_taps() {
            let timing = offset + tap.frame;
            // MP2K tracks are the MIDI channels; velocity is a 0..127 register, normalized here.
            let channel = tap.track as u8;
            context.send_event(match tap.velocity {
                Some(velocity) => NoteEvent::NoteOn {
                    timing,
                    voice_id: None,
                    channel,
                    note: tap.key,
                    velocity: f32::from(velocity) / 127.0,
                },
                None => NoteEvent::NoteOff {
                    timing,
                    voice_id: None,
                    channel,
                    note: tap.key,
                    velocity: 0.0,
                },
            });
        }
    }

    /// Picks up an engine the editor built, if one is waiting.
    ///
    /// `try_lock`, never `lock`: this runs on the audio thread, and the editor holds the mutex
    /// while it reads a ROM off disk. Missing a swap costs one buffer; blocking costs a dropout.
    fn take_pending_rom(&mut self) {
        if let Ok(mut pending) = self.rom.pending.try_lock()
            && let Some(controller) = pending.take()
        {
            self.controller = Some(controller);
        }
    }

    /// Swaps in the armed song engine (a [`GbaPlayer`]) and marks capture live.
    ///
    /// Called from `process` once the transport rolls from bar 1. Pulls the engine the background
    /// [`Task::ArmCapture`] parked in [`RomSlot::capture_engine`]; `try_lock` because `StopCapture`
    /// may be clearing it concurrently. `armed` is cleared either way so a lost race can't leave the
    /// plugin stuck armed; `active` only flips when an engine was actually there to start.
    fn begin_capture(&mut self) {
        let got = self
            .rom
            .capture_engine
            .try_lock()
            .ok()
            .and_then(|mut cap| cap.take());
        self.capture.armed.store(false, Ordering::SeqCst);
        if let Some(controller) = got {
            self.controller = Some(controller);
            self.capture.active.store(true, Ordering::SeqCst);
        }
    }

    /// The engine's parameter-driven player, if a ROM is loaded **and capture is not running**.
    ///
    /// The second half is load-bearing and easy to miss. During rip capture the controller holds a
    /// [`GbaPlayer`] instead, so this downcast returns `None` — and that single fact is what gates
    /// the whole mode: incoming MIDI notes, parameter pushes, and the host tempo all become no-ops,
    /// leaving the song's own bytecode authoritative over the registers being captured. Loosening
    /// this (e.g. returning some shared supertype) would silently let the DAW fight the song.
    fn player(&mut self) -> Option<&mut ParamPlayer> {
        self.controller
            .as_mut()?
            .player_mut()
            .as_any_mut()
            .downcast_mut::<ParamPlayer>()
    }

    /// Pushes every current parameter value into the engine.
    ///
    /// The engine diffs internally and raises MP2K's own `MPT_FLG_VOLCHG`/`PITCHG` for whatever
    /// actually moved, so pushing the whole set each block is cheap and keeps those ordering rules
    /// in one place instead of duplicating them here.
    fn push_params(&mut self) {
        let params = self.params.clone();
        let master = params.master_volume.value() as u8;
        let reverb = params.reverb.value() as u8;
        let max_chans = params.max_chans.value() as u8;
        let tracks: Vec<_> = params.tracks.iter().map(|t| t.to_engine()).collect();

        let Some(player) = self.player() else { return };
        player.set_master_volume(master);
        player.set_reverb(reverb);
        player.set_max_chans(max_chans);
        for (t, track) in tracks.iter().enumerate() {
            player.set_track_params(t, track);
        }
    }

    /// Renders `frames` stereo frames into `output` starting at `offset`.
    fn render(&mut self, output: &mut [&mut [f32]], offset: usize, frames: usize) {
        let Some(controller) = self.controller.as_mut() else {
            // No ROM: emit silence rather than leaving the host's buffer undefined.
            for ch in output.iter_mut() {
                ch[offset..offset + frames].fill(0.0);
            }
            return;
        };

        let scratch = &mut self.scratch[..frames * 2];
        controller.fill(scratch, &self.settings);
        for (i, frame) in scratch.chunks_exact(2).enumerate() {
            for (ch, &sample) in output.iter_mut().zip(frame) {
                ch[offset + i] = sample;
            }
        }
    }

    /// Applies one host note event to the engine.
    fn apply_note_event(&mut self, event: NoteEvent<()>) {
        let Some(player) = self.player() else { return };
        match event {
            NoteEvent::NoteOn {
                channel,
                note,
                velocity,
                ..
            } => {
                // MP2K velocity is a 0..127 register; nice-plug normalizes it.
                let v = (velocity * 127.0).round().clamp(0.0, 127.0) as u8;
                player.note_on(channel as usize, note, v);
            }
            // A choke is a hard stop, but MP2K's only way to end a note is its release, which is
            // what note-off does — so both take the same path.
            NoteEvent::NoteOff { channel, note, .. } | NoteEvent::Choke { channel, note, .. } => {
                player.note_off(channel as usize, note);
            }
            _ => {}
        }
    }
}

/// Whether the transport is at or very near the song start — bar 1.
///
/// Rip capture waits for this before starting the song, so the host's recording (which the user
/// also kicks off from bar 1) and the song begin on the same sample. Half a second of tolerance
/// comfortably covers the first buffer after a play-from-zero on any host while staying well clear
/// of any later bar the user might have scrubbed to; if the host reports no position at all (e.g.
/// the standalone wrapper), assume the start rather than never firing.
fn near_song_start(transport: &Transport) -> bool {
    const TOLERANCE_SECONDS: f64 = 0.5;
    match transport.pos_samples {
        Some(pos) => pos <= (f64::from(transport.sample_rate) * TOLERANCE_SECONDS) as i64,
        None => true,
    }
}

impl Plugin for OptimePlug {
    const NAME: &'static str = "Optime Player";
    const VENDOR: &'static str = "Optime Player";
    const URL: &'static str = "https://github.com/Powerlated/OptimePlayer";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: None,
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    /// Notes only. MP2K's control surface is the parameter list, not MIDI CCs — VST3 has no MIDI of
    /// its own anyway, and a host records parameter automation natively.
    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;

    /// Rip capture plays the song's bytecode and emits the notes it produces, so a host recording
    /// this plugin's output gets the song back as a clip. Silent outside capture.
    const MIDI_OUTPUT: MidiConfig = MidiConfig::Basic;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = Task;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    /// Runs [`Task`]s on a background thread — off both the audio and GUI threads.
    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let params = self.params.clone();
        let rom = self.rom.clone();
        let capture = self.capture.clone();
        Box::new(move |task| match task {
            Task::PickRom => rom.pick_and_load(&params),
            Task::SelectSong => {
                let song = *params.song_id.read().unwrap();
                rom.select_song(song, &params);
            }
            // Build and park the song engine *before* flagging armed, so `process` never sees armed
            // without a capture engine waiting for the transport trigger.
            Task::ArmCapture => {
                if rom.build_capture_engine(&params) {
                    capture.active.store(false, Ordering::SeqCst);
                    capture.armed.store(true, Ordering::SeqCst);
                }
            }
            Task::StopCapture => {
                let was_active = capture.active.swap(false, Ordering::SeqCst);
                capture.armed.store(false, Ordering::SeqCst);
                // Discard an armed-but-unstarted engine (built, parked, never swapped in).
                if let Ok(mut cap) = rom.capture_engine.try_lock() {
                    cap.take();
                }
                // Only rebuild the note-driven player if capture was actually running; otherwise the
                // ParamPlayer already in place is untouched and keeps the user's register edits.
                if was_active {
                    rom.rebuild(&params);
                }
            }
        })
    }

    fn editor(&mut self, async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.rom.clone(),
            self.capture.clone(),
            self.params.editor_state.clone(),
            async_executor,
        )
    }

    fn initialize(
        &mut self,
        _layout: &AudioIOLayout,
        config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = config.sample_rate as f64;
        self.scratch
            .resize(config.max_buffer_size as usize * 2, 0.0);
        // The editor builds engines too, and needs the output rate to do it.
        self.rom
            .sample_rate
            .store(config.sample_rate as u32, Ordering::Relaxed);
        self.load_rom();
        // A missing ROM is not a failed initialization — the user picks one later, and refusing here
        // would make the plugin un-instantiable before it could ever be given a file.
        true
    }

    fn reset(&mut self) {
        // Envelopes, channel allocation and LFO phase are all stateful, so a transport jump must
        // not leave a note ringing from the previous position. Silence every sounding note rather
        // than reloading the ROM — `reset` can be called from the audio thread, where file I/O is
        // not allowed.
        for t in 0..TRACKS {
            if let Some(player) = self.player() {
                player.all_notes_off(t);
            }
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Read the transport up front: its borrow of `context` must end before the block loop calls
        // `next_event`/`send_event`, which borrow `context` mutably.
        let transport = context.transport();
        let tempo = transport.tempo.unwrap_or(120.0);

        // Rip capture *arms*, then waits. The song does not start until the transport rolls from bar
        // 1, so the host's recording and the song begin on the same sample and the captured notes
        // land on bar 1 instead of wherever the playhead sat when "Arm capture" was clicked.
        if self.capture.armed.load(Ordering::Relaxed)
            && !self.capture.active.load(Ordering::Relaxed)
            && transport.playing
            && near_song_start(transport)
        {
            self.begin_capture();
        }

        self.take_pending_rom();

        let preset = self.params.preset.value();
        if preset != self.preset {
            self.preset = preset;
            self.settings = preset.settings();
        }

        // The host owns tempo; MP2K only needs it to pace the LFO and gate passes.
        if let Some(player) = self.player() {
            player.set_tempo_bpm(tempo);
        }
        self.push_params();

        let num_samples = buffer.samples();
        let output = buffer.as_slice();

        let mut next_event = context.next_event();
        let mut block_start = 0usize;
        while block_start < num_samples {
            // Apply every event landing at this instant, then cut the block at the next one, so a
            // note starts on its own sample rather than at the next block boundary.
            while let Some(event) = next_event {
                if event.timing() as usize > block_start {
                    break;
                }
                self.apply_note_event(event);
                next_event = context.next_event();
            }

            let next_at = next_event
                .map(|e| e.timing() as usize)
                .unwrap_or(num_samples);
            let block_end = next_at.min(block_start + MAX_BLOCK).min(num_samples);
            let frames = block_end - block_start;
            if frames == 0 {
                break;
            }
            self.render(output, block_start, frames);
            self.emit_captured_notes(context, block_start as u32);
            block_start = block_end;
        }

        self.publish_capture();
        ProcessStatus::Normal
    }
}

impl Vst3Plugin for OptimePlug {
    /// Fixed forever: a host rebinds a saved project's parameters through this.
    const VST3_CLASS_ID: [u8; 16] = *b"OptimePlayerMP2K";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Instrument,
        Vst3SubCategory::Synth,
        Vst3SubCategory::Stereo,
    ];
}

nice_export_vst3!(OptimePlug);

#[cfg(test)]
mod tests {
    use super::params::{OptimePlugParams, TRACKS, TrackParams};
    use nice_plug::prelude::Params;
    use optime_core::devices::gba::param_player::TrackParams as EngineTrackParams;

    /// A fresh plugin track must start exactly where a freshly started MP2K track does. The two
    /// default sets are written in different crates, so nothing but this stops them drifting — and
    /// if they drift, an untouched instance quietly stops sounding like the ROM.
    #[test]
    fn plugin_track_defaults_match_the_engines_track_reset() {
        assert_eq!(
            TrackParams::default().to_engine(),
            EngineTrackParams::default()
        );
    }

    /// VST3 rebinds a saved project's automation by parameter id, so the ids are a compatibility
    /// contract: every track needs its own, and two must never collide (a collision would silently
    /// merge two controls into one).
    #[test]
    fn every_track_gets_its_own_unique_parameter_ids() {
        let params = OptimePlugParams::default();
        let ids: Vec<String> = params
            .param_map()
            .into_iter()
            .map(|(id, _, _)| id)
            .collect();

        assert!(ids.contains(&"vol_1".to_string()), "track 1 volume");
        assert!(ids.contains(&"vol_16".to_string()), "track 16 volume");
        assert!(ids.contains(&"masvol".to_string()), "the global master");

        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "parameter ids must be unique");
        assert_eq!(
            ids.len(),
            22 * TRACKS + 4,
            "22 params per track plus masvol/reverb/maxchan/preset"
        );
    }
}
