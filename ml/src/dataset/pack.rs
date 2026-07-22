//! Multi-song **sequence packing** + window slicing over whole-song datasets.
//!
//! The real-song dataset stores each harvested song whole (intro+loop+loop,
//! variable length). Two consumers derive fixed-length training sequences from it:
//!
//! * **Packing** ([`pack_songs`]) — the long-context path (m03): several songs are
//!   concatenated into one `seq_len`-frame sequence, each followed by an **EOS
//!   slot**, tail padded. The layout is recorded in [`Song::doc_spans`]; the
//!   batcher turns it into per-slot kinds / document ids, the trunk into document
//!   masks and KDA state resets. Median song is 17 frames (SFX), so packing is
//!   what makes a 2048-frame window cheap instead of 99% pad.
//! * **Window slicing** ([`slice_into_windows`], [`sample_random_windows`]) — the
//!   fixed-window path (m00–m02 baselines): cut `seq_len`-frame windows exactly as
//!   the old harvest-time windowing did, now at load time.
//!
//! Packing is **re-done every epoch** with that epoch's RNG (different neighbours
//! each time — replaces random-offset window augmentation), and transposition
//! augmentation is applied **per constituent song** before concatenation (more
//! diverse than one shift per pack).

use crate::notes::{NoteEvent, Song};
use crate::theory::NO_CHORD;
use rand::seq::SliceRandom;
use rand::Rng;

/// Truncate a song to at most `max_frames` frames (notes clipped, labels/mask cut).
/// Returns the song unchanged when it already fits.
pub fn truncate_song(song: &Song, max_frames: usize) -> Song {
    if song.n_frames <= max_frames {
        return song.clone();
    }
    let mf = max_frames as u32;
    let notes = clip_notes_to_window(&song.notes, 0, mf);
    Song {
        n_frames: max_frames,
        notes,
        chord_labels: song.chord_labels[..max_frames.min(song.chord_labels.len())].to_vec(),
        label_mask: song
            .label_mask
            .as_ref()
            .map(|m| m[..max_frames.min(m.len())].to_vec()),
        loop_frame: song.loop_frame.filter(|&f| (f as usize) < max_frames),
        ..song.clone()
    }
}

/// Pack whole songs into `seq_len`-frame multi-document sequences.
///
/// Songs are shuffled, then first-fit packed (each song costs `n_frames + 1` slots
/// — its frames plus one EOS). Songs longer than `seq_len − 1` are truncated (the
/// harvest already truncates, so this is a belt-and-braces guard). Each pack is
/// materialised as one composite [`Song`] with `n_frames = seq_len`, notes offset
/// into place, `doc_spans` recording the layout, and `chord_labels` all
/// [`NO_CHORD`] (packs feed the unlabeled AR pretext only).
///
/// `augment` applies an independent random transposition to **each constituent**
/// before concatenation.
pub fn pack_songs<R: Rng>(songs: &[Song], seq_len: usize, augment: bool, rng: &mut R) -> Vec<Song> {
    assert!(seq_len >= 2, "seq_len must fit at least one frame + EOS");
    let mut order: Vec<usize> = (0..songs.len()).collect();
    order.shuffle(rng);

    // First-fit over the shuffled order: bins[i] = (used_slots, member indices).
    let mut bins: Vec<(usize, Vec<usize>)> = Vec::new();
    for &si in &order {
        let len = songs[si].n_frames.clamp(1, seq_len - 1);
        let cost = len + 1; // frames + EOS
        match bins.iter_mut().find(|(used, _)| used + cost <= seq_len) {
            Some((used, members)) => {
                *used += cost;
                members.push(si);
            }
            None => bins.push((cost, vec![si])),
        }
    }

    bins.iter()
        .map(|(_, members)| {
            let mut notes: Vec<NoteEvent> = Vec::new();
            let mut doc_spans = Vec::new();
            let mut cursor: u32 = 0;
            for &si in members {
                let shift = if augment {
                    crate::notes::random_transpose(rng)
                } else {
                    0
                };
                let s = truncate_song(&songs[si].transpose(shift), seq_len - 1);
                let len = s.n_frames as u32;
                for n in &s.notes {
                    notes.push(NoteEvent {
                        start_frame: n.start_frame + cursor,
                        end_frame: (n.end_frame + cursor).min(cursor + len),
                        ..*n
                    });
                }
                doc_spans.push((cursor, cursor + len));
                cursor += len + 1; // skip the EOS slot
            }
            Song {
                key_label: 0,
                n_frames: seq_len,
                notes,
                chord_labels: vec![NO_CHORD; seq_len],
                doc_spans,
                ..Song::default()
            }
        })
        .collect()
}

/// Notes overlapping `[win_start, win_end)`, clipped to the window and rebased to
/// its start. Empty if nothing sounds in the window.
///
/// Public because the labelled-window path ([`crate::annotations`], `eval_labeled`)
/// has to cut the exact same windows this module's own slicing does — two different
/// notions of "window" would silently misalign notes against their labels.
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
            ..Song::default()
        });
    }
    songs
}

/// Sample **random-offset overlapping windows** from a full-song note stream (the
/// fixed-window generations' pretraining augmentation — packing's counterpart for
/// backbones that can't take variable length). Each window starts at a uniformly
/// random offset in `[0, total - seq_len]`; yields roughly `coverage ×` the fixed
/// tiling count.
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
            ..Song::default()
        });
    }
    songs
}

/// Slice every whole song in a dataset into fixed windows (load-time windowing for
/// the fixed-window generations, replacing the old harvest-time windowing). Songs
/// shorter than `seq_len` are dropped.
pub fn window_dataset(songs: &[Song], seq_len: usize) -> Vec<Song> {
    songs
        .iter()
        .flat_map(|s| {
            let mut windows = slice_into_windows(&s.notes, seq_len, s.is_music);
            for w in windows.iter_mut() {
                w.source = s.source.clone();
            }
            windows
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::Instrument;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn song(n_frames: usize, pitch: u8) -> Song {
        Song {
            key_label: 3,
            n_frames,
            notes: (0..n_frames as u32)
                .map(|f| NoteEvent {
                    start_frame: f,
                    end_frame: f + 1,
                    pitch,
                    velocity: 0.8,
                    instrument: Instrument::Harmony,
                    track: 1,
                    pan: 0.0,
                })
                .collect(),
            chord_labels: vec![0; n_frames],
            ..Song::default()
        }
    }

    #[test]
    fn packs_songs_with_eos_and_pad() {
        let songs = vec![song(10, 60), song(5, 62), song(3, 64)];
        let mut rng = StdRng::seed_from_u64(1);
        let packs = pack_songs(&songs, 32, false, &mut rng);
        // 10+1 + 5+1 + 3+1 = 21 <= 32 -> one pack.
        assert_eq!(packs.len(), 1);
        let p = &packs[0];
        assert_eq!(p.n_frames, 32);
        assert_eq!(p.doc_spans.len(), 3);
        // Spans tile with one EOS slot between: end + 1 == next start.
        for w in p.doc_spans.windows(2) {
            assert_eq!(w[0].1 + 1, w[1].0, "EOS slot between docs");
        }
        // Every note lands inside its own doc's span.
        for n in &p.notes {
            assert!(
                p.doc_spans
                    .iter()
                    .any(|&(s, e)| n.start_frame >= s && n.end_frame <= e),
                "note {}..{} escapes all doc spans",
                n.start_frame,
                n.end_frame
            );
        }
        // Total real frames preserved.
        assert_eq!(p.real_frames(), 18);
    }

    #[test]
    fn packing_is_deterministic_from_seed_and_reshuffles() {
        let songs: Vec<Song> = (0..20).map(|i| song(5 + (i % 7), 60 + i as u8)).collect();
        let a = pack_songs(&songs, 64, false, &mut StdRng::seed_from_u64(7));
        let b = pack_songs(&songs, 64, false, &mut StdRng::seed_from_u64(7));
        let c = pack_songs(&songs, 64, false, &mut StdRng::seed_from_u64(8));
        let spans = |ps: &[Song]| -> Vec<Vec<(u32, u32)>> {
            ps.iter().map(|p| p.doc_spans.clone()).collect()
        };
        let notes = |ps: &[Song]| -> Vec<usize> { ps.iter().map(|p| p.notes.len()).collect() };
        assert_eq!(spans(&a), spans(&b), "same seed, same packs");
        assert!(
            spans(&a) != spans(&c) || notes(&a) != notes(&c),
            "different seed should repack"
        );
    }

    #[test]
    fn oversized_song_is_truncated_not_dropped() {
        let songs = vec![song(100, 60)];
        let mut rng = StdRng::seed_from_u64(2);
        let packs = pack_songs(&songs, 16, false, &mut rng);
        assert_eq!(packs.len(), 1);
        assert_eq!(packs[0].doc_spans, vec![(0, 15)]); // truncated to seq_len-1
        assert!(packs[0].notes.iter().all(|n| n.end_frame <= 15));
    }

    #[test]
    fn window_dataset_slices_and_stamps_source() {
        let mut s = song(20, 60);
        s.source = "rom-a".into();
        let windows = window_dataset(&[s], 8);
        assert_eq!(windows.len(), 2); // 20/8 = 2 full windows
        assert!(windows.iter().all(|w| w.source == "rom-a"));
        assert!(windows.iter().all(|w| w.n_frames == 8));
    }
}
