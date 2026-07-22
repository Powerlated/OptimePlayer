//! Real game-song harvesting (feature `harvest`).
//!
//! Runs a device sequencer **headlessly** — the same trick
//! [`optime_core::FsVisController`] / `SongOverview::overview` use for the
//! visualizer — and turns its `SynthEvent` stream into unlabeled ml [`Song`]s on
//! the *same* 4-frames-per-beat grid the synthetic generator uses ([`notes`]).
//!
//! These songs carry no chord/key ground truth (`chord_labels` are all
//! [`NO_CHORD`]); they exist to teach the encoder the *real* note-event
//! distribution during self-supervised pretraining. The chord/key label mapping
//! still comes from the synthetic set at fine-tune time.
//!
//! [`notes`]: crate::notes

use crate::notes::{Instrument, NoteEvent, Song, FRAMES_PER_BEAT};
use crate::theory::NO_CHORD;
// The windowing helpers moved to [`crate::pack`] (no engine dependency, so fixed-
// window backbones can window whole-song datasets at load time); re-exported here
// for the label-window cutters that already import them from `harvest`.
pub use crate::pack::{clip_notes_to_window, sample_random_windows, slice_into_windows};
use optime_core::devices::gba::GbaRom;
use optime_core::synth_controller::messages::TickFeedback;
use optime_core::{load_all, PerDeviceSettings, SoundData, SynthEvent, VoiceId};
use std::collections::{HashMap, HashSet};

/// Weak is-music labels: game code → set of curated (real-music) song ids. Built
/// from the app's `song_names/*.json` via [`Annotations::from_files`]. A GBA song
/// whose id is in the set is `Some(true)`; any other id under an annotated code is
/// `Some(false)` (an unlisted SFX/jingle table entry); everything else is `None`.
#[derive(Debug, Clone, Default)]
pub struct Annotations {
    music_ids: HashMap<String, HashSet<u32>>,
}

impl Annotations {
    /// Load `(game_code, path)` pairs of `song_names` JSON files
    /// (`[{"songId":N,"title":"…"}]`). Unreadable/unparseable files are skipped
    /// with a warning.
    pub fn from_files<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a std::path::Path)>) -> Self {
        #[derive(serde::Deserialize)]
        struct Entry {
            #[serde(rename = "songId")]
            song_id: u32,
        }
        let mut music_ids: HashMap<String, HashSet<u32>> = HashMap::new();
        for (code, path) in pairs {
            match std::fs::read_to_string(path)
                .ok()
                .and_then(|s| serde_json::from_str::<Vec<Entry>>(&s).ok())
            {
                Some(entries) => {
                    let ids = music_ids.entry(code.to_string()).or_default();
                    for e in entries {
                        ids.insert(e.song_id);
                    }
                }
                None => eprintln!(
                    "  (skipping annotation {code}={} — unreadable)",
                    path.display()
                ),
            }
        }
        Annotations { music_ids }
    }

    /// Weak is-music label for one song id under `game_code`.
    fn label(&self, game_code: Option<&str>, id: u32) -> Option<bool> {
        let ids = self.music_ids.get(game_code?)?;
        Some(ids.contains(&id))
    }
}

/// Safety cap so a song that neither loops nor ends can't spin forever (matches
/// the cap `SongOverview::overview` uses).
const MAX_STEPS: u32 = 200_000;

/// Convert a sequencer-step timestamp to a frame index on the ml grid.
///
/// The ml grid is [`FRAMES_PER_BEAT`] frames per beat; the device reports its own
/// `steps_per_beat` (DS SSEQ: 48, GBA MP2K: 24), so `frame = step * FPB /
/// steps_per_beat`. `steps_per_beat <= 0` (device reports no beat) falls back to
/// treating one step as one frame.
pub fn step_to_frame(step: u32, steps_per_beat: f64) -> u32 {
    if steps_per_beat > 0.0 {
        ((step as f64) * FRAMES_PER_BEAT as f64 / steps_per_beat).round() as u32
    } else {
        step
    }
}

/// A note being built up as the stream plays: opened at `NoteStarted`, closed at
/// the matching `NoteReleased` / `VoiceStopped`. `velocity` tracks the **peak**
/// per-tick volume seen while the note is open, because envelopes start at 0 and
/// ramp through attack — the note-on volume is often 0 and would zero the chroma.
struct OpenNote {
    voice: VoiceId,
    track: usize,
    key: u8,
    start_frame: u32,
    velocity: f32,
    pan: f32,
}

/// Harvest every playable song from one loaded archive as a **whole** [`Song`]
/// (intro + loop + loop, variable `n_frames`).
///
/// When `data` is a GBA ROM whose game code is present in `annotations`, each
/// song is stamped with a weak is-music label; otherwise songs are unlabeled.
pub fn harvest_sound_data_full(data: &dyn SoundData, annotations: &Annotations) -> Vec<Song> {
    // Game code (GBA only) selects the annotation set; DS/DSE stay unlabeled.
    let game_code = data
        .as_any()
        .downcast_ref::<GbaRom>()
        .and_then(|rom| rom.game_code());

    let mut out = Vec::new();
    for id in data.song_ids() {
        let is_music = annotations.label(game_code.as_deref(), id);
        if let Some(h) = harvest_song(data, id) {
            if !h.notes.is_empty() {
                let n_frames = h.notes.iter().map(|n| n.end_frame).max().unwrap_or(0) as usize;
                if n_frames == 0 {
                    continue;
                }
                out.push(Song {
                    key_label: 0, // unlabeled sentinel
                    n_frames,
                    notes: h.notes,
                    chord_labels: vec![NO_CHORD; n_frames],
                    is_music,
                    loop_frame: h.loop_frame,
                    ..Song::default()
                });
            }
        }
    }
    out
}

/// One harvested song's raw stream: note events, the device's `steps_per_beat`
/// (so callers can map ml frames back to sequencer steps), and the frame of the
/// first loop point when the song looped.
pub struct HarvestedSong {
    pub notes: Vec<NoteEvent>,
    pub steps_per_beat: f64,
    pub loop_frame: Option<u32>,
}

/// Run one song headlessly and collect its full un-windowed note-event stream.
/// Returns `None` if the song can't be started or is silent.
pub fn harvest_song_full(data: &dyn SoundData, id: u32) -> Option<HarvestedSong> {
    harvest_song(data, id)
}

/// Run one song headlessly and collect its full note-event stream (no windowing).
///
/// A looping song is captured as **intro + loop + loop** — the run stops at the
/// *second* [`SynthEvent::Looped`] — so the model can train on the music crossing
/// a loop boundary. A non-looping song stops at [`SynthEvent::Ended`]. Returns
/// `None` if the song can't be started.
fn harvest_song(data: &dyn SoundData, id: u32) -> Option<HarvestedSong> {
    let mut player = data.make_player(id)?;
    let steps_per_beat = player.steps_per_beat();
    let config = PerDeviceSettings::neutral();
    let mut feedback = TickFeedback::default();
    let mut events: Vec<SynthEvent> = Vec::new();

    let mut open: Vec<OpenNote> = Vec::new();
    let mut finished: Vec<NoteEvent> = Vec::new();
    // Latest pan per track (from `TrackPan`), applied to notes at their onset.
    let mut track_pan: Vec<f32> = Vec::new();
    let mut loop_frame: Option<u32> = None;

    loop {
        events.clear();
        player.tick(&mut feedback, &config, &mut events);
        let now = step_to_frame(player.steps_elapsed(), steps_per_beat);
        let mut stop = false;

        for ev in events.drain(..) {
            match ev {
                SynthEvent::NoteStarted {
                    voice,
                    track,
                    key,
                    volume,
                    ..
                } => {
                    let pan = track_pan.get(track).copied().unwrap_or(0.0);
                    open.push(OpenNote {
                        voice,
                        track,
                        key,
                        start_frame: now,
                        velocity: volume as f32,
                        pan,
                    });
                }
                SynthEvent::VoiceVolume { voice, volume, .. } => {
                    // Track the note's peak volume (envelopes ramp up from 0).
                    if let Some(n) = open.iter_mut().find(|n| n.voice == voice) {
                        n.velocity = n.velocity.max(volume as f32);
                    }
                }
                SynthEvent::NoteReleased { track, key } => {
                    close_note(&mut open, &mut finished, track, key, now);
                }
                SynthEvent::VoiceStopped { voice, .. } => {
                    // Close the note owning this voice (envelope floor / steal ended
                    // a note that never signalled a release).
                    close_voice(&mut open, &mut finished, voice, now);
                }
                SynthEvent::TrackPan {
                    track,
                    pan_vol_l,
                    pan_vol_r,
                } => {
                    if track >= track_pan.len() {
                        track_pan.resize(track + 1, 0.0);
                    }
                    // Normalized split (~sums to 1) → signed pan in [-1, 1].
                    track_pan[track] = (pan_vol_r - pan_vol_l) as f32;
                }
                // Intro + loop + loop: keep playing through the first loop point
                // (recording where it was) and stop at the second.
                SynthEvent::Looped => match loop_frame {
                    None => loop_frame = Some(now),
                    Some(_) => stop = true,
                },
                SynthEvent::Ended => stop = true,
                _ => {}
            }
        }

        if stop || now >= MAX_STEPS {
            // Flush anything still sounding to the loop/end boundary.
            for n in open.drain(..) {
                push_note(&mut finished, n, now);
            }
            break;
        }
    }

    if finished.is_empty() {
        None
    } else {
        Some(HarvestedSong {
            notes: finished,
            steps_per_beat,
            loop_frame,
        })
    }
}

fn close_note(
    open: &mut Vec<OpenNote>,
    finished: &mut Vec<NoteEvent>,
    track: usize,
    key: u8,
    now: u32,
) {
    if let Some(i) = open.iter().position(|n| n.track == track && n.key == key) {
        let n = open.swap_remove(i);
        push_note(finished, n, now);
    }
}

fn close_voice(open: &mut Vec<OpenNote>, finished: &mut Vec<NoteEvent>, voice: VoiceId, now: u32) {
    if let Some(i) = open.iter().position(|n| n.voice == voice) {
        let n = open.swap_remove(i);
        push_note(finished, n, now);
    }
}

/// Finalize an open note into a `NoteEvent`, giving zero-length notes a single
/// frame so `end_frame > start_frame` always holds.
///
/// Every harvested note is treated as pitched ([`Instrument::Harmony`]): the
/// `SynthEvent` stream exposes no voice kind, and the only cheap proxy (note
/// duration) misfired badly — it labelled ~70% of real notes (fast melody, arps,
/// staccato bass) as percussion, zeroing the harmonic chroma. Letting a little
/// drum noise into the chroma is far better than erasing the harmony. Reliable
/// percussion detection needs a voice-kind signal (PSG noise vs. sample) and is
/// left as a future refinement.
fn push_note(finished: &mut Vec<NoteEvent>, n: OpenNote, end_frame: u32) {
    let end = end_frame.max(n.start_frame + 1);
    finished.push(NoteEvent {
        start_frame: n.start_frame,
        end_frame: end,
        pitch: n.key,
        velocity: n.velocity,
        instrument: Instrument::Harmony,
        // The real per-note channel — the only instrument cue that survives
        // harvesting (role is unknown). Clamp into the 0..15 token range.
        track: n.track.min(15) as u8,
        pan: n.pan,
    });
}

/// Harvest every archive in a directory into whole [`Song`]s (intro+loop+loop,
/// variable length), stamped with the archive's file stem as `Song::source` and
/// weak is-music labels for covered GBA game codes. Files that don't parse into
/// any `SoundData` are silently skipped.
pub fn harvest_dir_full(
    dir: &std::path::Path,
    annotations: &Annotations,
) -> std::io::Result<Vec<Song>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        // Archive file stem = the song's `Song::source` (per-ROM data breakdown).
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        for data in load_all(&bytes) {
            let mut songs = harvest_sound_data_full(data.as_ref(), annotations);
            if songs.is_empty() {
                continue;
            }
            let music = songs.iter().filter(|s| s.is_music == Some(true)).count();
            let sfx = songs.iter().filter(|s| s.is_music == Some(false)).count();
            let looped = songs.iter().filter(|s| s.loop_frame.is_some()).count();
            eprintln!(
                "  {}: {} songs ({} music / {} sfx / {} unlabeled; {} looped)",
                path.file_name().unwrap_or_default().to_string_lossy(),
                songs.len(),
                music,
                sfx,
                songs.len() - music - sfx,
                looped,
            );
            for s in songs.iter_mut() {
                s.source = stem.clone();
            }
            out.extend(songs);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_to_frame_matches_beat_grids() {
        // DS: 48 steps/beat, 4 frames/beat -> 12 steps per frame.
        assert_eq!(step_to_frame(0, 48.0), 0);
        assert_eq!(step_to_frame(12, 48.0), 1);
        assert_eq!(step_to_frame(48, 48.0), 4); // one beat = 4 frames
                                                // GBA: 24 steps/beat -> 6 steps per frame.
        assert_eq!(step_to_frame(6, 24.0), 1);
        assert_eq!(step_to_frame(24, 24.0), 4);
        // Degenerate: no beat division -> step == frame.
        assert_eq!(step_to_frame(7, 0.0), 7);
    }

    #[test]
    fn annotations_label_music_vs_sfx() {
        let mut music_ids = HashMap::new();
        music_ids.insert("BPEE".to_string(), HashSet::from([405u32, 413]));
        let ann = Annotations { music_ids };
        // Curated id under the annotated code → music; other id → sfx.
        assert_eq!(ann.label(Some("BPEE"), 405), Some(true));
        assert_eq!(ann.label(Some("BPEE"), 999), Some(false));
        // Unknown code / no code → unlabeled.
        assert_eq!(ann.label(Some("A3UJ"), 405), None);
        assert_eq!(ann.label(None, 405), None);
    }
}
