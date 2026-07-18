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
use optime_core::devices::gba::GbaRom;
use optime_core::synth_controller::messages::TickFeedback;
use optime_core::{load_all, PerDeviceSettings, SoundData, SynthEvent, VoiceId};
use rand::Rng;
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

/// Harvest every playable song from one loaded archive into [`Song`] windows of
/// exactly `seq_len` frames (longer songs are sliced into consecutive windows;
/// the trailing partial window is dropped).
///
/// When `data` is a GBA ROM whose game code is present in `annotations`, each
/// window is stamped with a weak is-music label; otherwise windows are unlabeled.
pub fn harvest_sound_data(
    data: &dyn SoundData,
    seq_len: usize,
    annotations: &Annotations,
) -> Vec<Vec<Song>> {
    harvest_sound_data_full(data, annotations)
        .into_iter()
        .map(|(notes, is_music)| slice_into_windows(&notes, seq_len, is_music))
        .filter(|windows| !windows.is_empty())
        .collect()
}

/// Harvest every playable song as its full, **un-windowed** note-event stream +
/// weak is-music label (one entry per source song). Windowing is left to the
/// caller, so the train/val split can window differently — e.g. random-offset
/// overlapping windows for train (more, phase-diverse data) vs fixed tiling for a
/// clean, non-redundant val set.
pub fn harvest_sound_data_full(
    data: &dyn SoundData,
    annotations: &Annotations,
) -> Vec<(Vec<NoteEvent>, Option<bool>)> {
    // Game code (GBA only) selects the annotation set; DS/DSE stay unlabeled.
    let game_code = data
        .as_any()
        .downcast_ref::<GbaRom>()
        .and_then(|rom| rom.game_code());

    let mut out = Vec::new();
    for id in data.song_ids() {
        let is_music = annotations.label(game_code.as_deref(), id);
        if let Some((notes, _steps_per_beat)) = harvest_song(data, id) {
            if !notes.is_empty() {
                out.push((notes, is_music));
            }
        }
    }
    out
}

/// Run one song headlessly and collect its full un-windowed note-event stream
/// plus the device's `steps_per_beat` (so callers can map ml frames back to
/// sequencer steps). Returns `None` if the song can't be started or is silent.
pub fn harvest_song_full(data: &dyn SoundData, id: u32) -> Option<(Vec<NoteEvent>, f64)> {
    harvest_song(data, id)
}

/// Run one song headlessly and collect its full note-event stream (no windowing)
/// and the device's `steps_per_beat`. Returns `None` if the song can't be started.
fn harvest_song(data: &dyn SoundData, id: u32) -> Option<(Vec<NoteEvent>, f64)> {
    let mut player = data.make_player(id)?;
    let steps_per_beat = player.steps_per_beat();
    let config = PerDeviceSettings::neutral();
    let mut feedback = TickFeedback::default();
    let mut events: Vec<SynthEvent> = Vec::new();

    let mut open: Vec<OpenNote> = Vec::new();
    let mut finished: Vec<NoteEvent> = Vec::new();
    // Latest pan per track (from `TrackPan`), applied to notes at their onset.
    let mut track_pan: Vec<f32> = Vec::new();

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
                SynthEvent::Looped | SynthEvent::Ended => stop = true,
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
        Some((finished, steps_per_beat))
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

/// Notes overlapping `[win_start, win_end)`, clipped to the window and rebased to
/// its start. Empty if nothing sounds in the window.
/// Clip a note stream to `[win_start, win_end)`, rebasing frame indices to the window start.
///
/// Public because the labelled-window path ([`crate::annotations`], `eval_labeled`) has to cut the
/// exact same windows this module's own slicing does — two different notions of "window" would
/// silently misalign notes against their labels.
pub fn clip_notes_to_window(notes: &[NoteEvent], win_start: u32, win_end: u32) -> Vec<NoteEvent> {
    let mut out = Vec::new();
    for n in notes {
        if n.end_frame <= win_start || n.start_frame >= win_end {
            continue;
        }
        let start = n.start_frame.max(win_start) - win_start;
        let end = n.end_frame.min(win_end) - win_start;
        if end > start {
            out.push(NoteEvent {
                start_frame: start,
                end_frame: end,
                ..*n
            });
        }
    }
    out
}

/// Slice a full-song note stream into consecutive `seq_len`-frame windows, each a
/// standalone [`Song`] with frame indices rebased to the window start and the
/// song's weak is-music label. The trailing partial window (and any window with
/// no notes) is dropped.
pub fn slice_into_windows(
    notes: &[NoteEvent],
    seq_len: usize,
    is_music: Option<bool>,
) -> Vec<Song> {
    if seq_len == 0 || notes.is_empty() {
        return Vec::new();
    }
    let total_frames = notes.iter().map(|n| n.end_frame).max().unwrap_or(0) as usize;
    let n_windows = total_frames / seq_len;
    let sl = seq_len as u32;

    let mut songs = Vec::with_capacity(n_windows);
    for w in 0..n_windows {
        let win_start = (w as u32) * sl;
        let win_notes = clip_notes_to_window(notes, win_start, win_start + sl);
        if win_notes.is_empty() {
            continue;
        }
        songs.push(Song {
            key_label: 0, // unlabeled sentinel; unused during pretraining
            n_frames: seq_len,
            notes: win_notes,
            chord_labels: vec![NO_CHORD; seq_len],
            is_music,
        });
    }
    songs
}

/// Sample **random-offset overlapping windows** from a full-song note stream, for
/// pretraining augmentation: each window starts at a uniformly random offset in
/// `[0, total - seq_len]`, so the encoder sees phrase context beginning at every
/// beat phase instead of only at fixed `seq_len` boundaries. Yields roughly
/// `coverage ×` the fixed-tiling window count (`coverage ≥ 1`), multiplying the
/// pretraining set; the AR next-frame pretext has no beat-phase assumption, so
/// random offsets are safe (the beat-aware smoothness term runs only at fine-tune,
/// on synthetic data, which stays a single window). Notes are clipped/rebased per
/// window exactly like [`slice_into_windows`].
pub fn sample_random_windows<R: Rng>(
    notes: &[NoteEvent],
    seq_len: usize,
    coverage: f64,
    is_music: Option<bool>,
    rng: &mut R,
) -> Vec<Song> {
    if seq_len == 0 || notes.is_empty() {
        return Vec::new();
    }
    let total = notes.iter().map(|n| n.end_frame).max().unwrap_or(0) as usize;
    if total < seq_len {
        return Vec::new(); // can't fit even one window
    }
    let max_offset = total - seq_len; // inclusive upper bound for a start offset
    let tiled = (total / seq_len).max(1);
    let target = ((tiled as f64) * coverage.max(1.0)).ceil() as usize;
    let sl = seq_len as u32;

    let mut songs = Vec::with_capacity(target);
    for _ in 0..target {
        let off = rng.gen_range(0..=max_offset) as u32;
        let win_notes = clip_notes_to_window(notes, off, off + sl);
        if win_notes.is_empty() {
            continue;
        }
        songs.push(Song {
            key_label: 0,
            n_frames: seq_len,
            notes: win_notes,
            chord_labels: vec![NO_CHORD; seq_len],
            is_music,
        });
    }
    songs
}

/// Harvest every archive in a directory of ROMs/audio dumps into [`Song`]s,
/// applying `annotations` (weak is-music labels) to GBA ROMs whose game code is
/// covered. Files that don't parse into any `SoundData` are silently skipped.
pub fn harvest_dir(
    dir: &std::path::Path,
    seq_len: usize,
    annotations: &Annotations,
) -> std::io::Result<Vec<Vec<Song>>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        for data in load_all(&bytes) {
            let songs = harvest_sound_data(data.as_ref(), seq_len, annotations);
            let n_windows: usize = songs.iter().map(|s| s.len()).sum();
            if n_windows > 0 {
                let flat = songs.iter().flatten();
                let music = flat.clone().filter(|s| s.is_music == Some(true)).count();
                let sfx = flat.filter(|s| s.is_music == Some(false)).count();
                eprintln!(
                    "  {}: {} songs, {} windows ({} music / {} sfx / {} unlabeled)",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    songs.len(),
                    n_windows,
                    music,
                    sfx,
                    n_windows - music - sfx,
                );
                out.extend(songs);
            }
        }
    }
    Ok(out)
}

/// Harvest every archive in a directory into its per-song **un-windowed** note
/// stream + weak is-music label (the full-stream counterpart to [`harvest_dir`]).
/// Lets the caller window train/val differently after the song-level split.
pub fn harvest_dir_full(
    dir: &std::path::Path,
    annotations: &Annotations,
) -> std::io::Result<Vec<(Vec<NoteEvent>, Option<bool>)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        for data in load_all(&bytes) {
            out.extend(harvest_sound_data_full(data.as_ref(), annotations));
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
    fn windows_rebase_and_drop_partial() {
        // Two notes: one in window 0, one straddling into window 1.
        let notes = vec![
            NoteEvent {
                start_frame: 2,
                end_frame: 6,
                pitch: 60,
                velocity: 0.8,
                instrument: Instrument::Harmony,
                track: 0,
                pan: 0.0,
            },
            NoteEvent {
                start_frame: 6,
                end_frame: 14,
                pitch: 64,
                velocity: 0.8,
                instrument: Instrument::Harmony,
                track: 0,
                pan: 0.0,
            },
        ];
        // seq_len 8, total_frames 14 -> exactly one full window (frames 0..8).
        let songs = slice_into_windows(&notes, 8, Some(true));
        assert_eq!(songs.len(), 1);
        let s = &songs[0];
        assert_eq!(s.n_frames, 8);
        assert_eq!(s.chord_labels, vec![NO_CHORD; 8]);
        assert_eq!(s.is_music, Some(true)); // label propagates to windows
                                            // First note verbatim; second clipped to the window boundary (6..8).
        assert!(s
            .notes
            .iter()
            .any(|n| n.start_frame == 2 && n.end_frame == 6));
        assert!(s
            .notes
            .iter()
            .any(|n| n.start_frame == 6 && n.end_frame == 8));
    }

    #[test]
    fn sample_random_windows_oversamples_and_stays_in_bounds() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        // 4 windows worth of notes (total 32 frames) at seq_len 8 -> tiled count 4.
        // Pitch varies by frame so different offsets capture different pitch sets
        // (otherwise uniform notes look identical once rebased to the window).
        let notes: Vec<NoteEvent> = (0..32u32)
            .map(|f| NoteEvent {
                start_frame: f,
                end_frame: f + 2,
                pitch: 60 + f as u8,
                velocity: 0.8,
                instrument: Instrument::Harmony,
                track: 0,
                pan: 0.0,
            })
            .collect();
        let mut rng = StdRng::seed_from_u64(1);
        let songs = sample_random_windows(&notes, 8, 4.0, Some(true), &mut rng);
        // coverage 4 × tiled 4 = 16 windows.
        assert_eq!(songs.len(), 16);
        let mut signatures: HashSet<Vec<u8>> = HashSet::new();
        for s in &songs {
            assert_eq!(s.n_frames, 8);
            assert_eq!(s.chord_labels, vec![NO_CHORD; 8]);
            assert_eq!(s.is_music, Some(true));
            // Every note rebased into [0, 8).
            assert!(s
                .notes
                .iter()
                .all(|n| n.start_frame < 8 && n.end_frame <= 8));
            let mut pitches: Vec<u8> = s.notes.iter().map(|n| n.pitch).collect();
            pitches.sort();
            signatures.insert(pitches);
        }
        // Phase diversity: random start offsets must yield >1 distinct window
        // (pitch sets differ because each offset captures different frames).
        assert!(
            signatures.len() > 1,
            "random offsets must yield phase-diverse windows"
        );
    }

    #[test]
    fn sample_random_windows_drops_song_shorter_than_window() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        // total 5 frames < seq_len 8 -> no window fits.
        let notes = vec![NoteEvent {
            start_frame: 0,
            end_frame: 5,
            pitch: 60,
            velocity: 0.8,
            instrument: Instrument::Harmony,
            track: 0,
            pan: 0.0,
        }];
        let mut rng = StdRng::seed_from_u64(2);
        assert!(sample_random_windows(&notes, 8, 4.0, None, &mut rng).is_empty());
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
