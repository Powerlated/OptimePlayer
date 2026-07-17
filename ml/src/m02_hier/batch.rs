//! Generation 02's pooling input: each frame's variable-length set of **sounding** notes
//! (onset..onset+duration), capped at [`MAX_POLY`]. A held note is a member of every frame it
//! sounds through, flagged attack-vs-held — contrast [`crate::m01_event`], which pools only the
//! *onsetting* notes and leans on `dur` for sustain.

use crate::notes::Song;
use crate::tokenize::{self, EventExample, NoteToken};

/// Max notes pooled per frame (the set-transformer sequence length). Capped below
/// the measured max simultaneous polyphony (22, per `token_stats`) as a speed/memory
/// tradeoff: frames with >16 simultaneous notes drop the extras, but those are rare,
/// and the smaller set grid (`[batch*n_frames, 17, d]` vs 25) cuts the dominant
/// set-attention cost substantially. Tweakable per run.
pub const MAX_POLY: usize = 16;

/// Grouped batch held as a **flat list of real (note, sounding-frame) entries**
/// (`n_snd` long), not a dense `[batch*n_frames*MAX_POLY]` grid. The model embeds
/// the field indices over this short list (cheap) and scatter-adds each into its
/// `(frame, position)` slot to materialise the set grid only once — avoiding the
/// padded-grid field-sum that dominated allocations in the dense version.
pub struct HierBatchData {
    pub batch: usize,
    pub n_frames: usize,
    /// Number of real (note, sounding-frame) entries — the flat note-list length.
    pub n_snd: usize,
    pub pitch: Vec<i64>,
    pub pc: Vec<i64>,
    pub channel: Vec<i64>,
    pub vel: Vec<i64>,
    pub pan: Vec<i64>,
    pub dur: Vec<i64>,
    pub role: Vec<i64>,
    /// 1 if the note *onsets* in this frame, else 0 (attack vs. held).
    pub onset: Vec<i64>,
    /// Destination slot for each entry = `frame_row * MAX_POLY + position`
    /// (`frame_row = example*n_frames + frame`), into the `[batch*n_frames, MAX_POLY, d]`
    /// grid built by scatter-add. Each slot is written at most once.
    pub slot_row: Vec<i64>,
    /// Pad mask for the `[batch*n_frames, MAX_POLY + 1]` set, row-major (col 0 =
    /// CLS, always valid). `true` = empty slot, ignored by set-attention.
    /// Precomputed at batch build so the forward path no longer rebuilds it.
    pub pad_mask: Vec<bool>,
    pub chord_labels: Vec<usize>,
    pub key_labels: Vec<usize>,
    /// Retained so the AR pretext can rebuild per-frame sounding content.
    pub examples: Vec<EventExample>,
}

impl HierBatchData {
    /// Build from songs (all must share `n_frames`).
    pub fn build(songs: &[Song]) -> HierBatchData {
        Self::from_examples(tokenize::examples(songs))
    }

    /// Build from already-tokenized examples.
    pub fn from_examples(examples: Vec<EventExample>) -> HierBatchData {
        let batch = examples.len();
        let nf = examples.first().map(|e| e.n_frames).unwrap_or(0);
        let bnf = batch * nf;
        let stride = MAX_POLY + 1;

        let cap: usize = examples
            .iter()
            .map(|e| {
                e.tokens
                    .iter()
                    .map(|t| t.dur_frames.min(nf as u32) as usize)
                    .sum::<usize>()
            })
            .sum();
        let mut d = HierBatchData {
            batch,
            n_frames: nf,
            n_snd: 0,
            pitch: Vec::with_capacity(cap),
            pc: Vec::with_capacity(cap),
            channel: Vec::with_capacity(cap),
            vel: Vec::with_capacity(cap),
            pan: Vec::with_capacity(cap),
            dur: Vec::with_capacity(cap),
            role: Vec::with_capacity(cap),
            onset: Vec::with_capacity(cap),
            slot_row: Vec::with_capacity(cap),
            // Default: every slot padded (true). CLS col + filled note cols cleared below.
            pad_mask: vec![true; bnf * stride],
            chord_labels: Vec::with_capacity(batch * nf),
            key_labels: Vec::with_capacity(batch),
            examples: Vec::new(),
        };
        for (bi, ex) in examples.iter().enumerate() {
            // Bucket sounding notes per frame (onset..onset+dur).
            let mut per_frame: Vec<Vec<&NoteToken>> = vec![Vec::new(); nf];
            for t in &ex.tokens {
                let end = ((t.onset + t.dur_frames) as usize).min(nf);
                for slot in per_frame[(t.onset as usize)..end].iter_mut() {
                    if slot.len() < MAX_POLY {
                        slot.push(t);
                    }
                }
            }
            for (f, notes) in per_frame.iter().enumerate() {
                let frame_row = bi * nf + f;
                let base = frame_row * stride;
                // CLS (col 0) always participates.
                d.pad_mask[base] = false;
                for (pos, t) in notes.iter().enumerate() {
                    d.pitch.push(t.pitch);
                    d.pc.push(t.pc);
                    d.channel.push(t.channel);
                    d.vel.push(t.vel);
                    d.pan.push(t.pan);
                    d.dur.push(t.dur);
                    d.role.push(t.role);
                    d.onset.push((t.onset as usize == f) as i64);
                    d.slot_row.push((frame_row * MAX_POLY + pos) as i64);
                    // Note `pos` sits at set column `pos + 1` (after the prepended CLS).
                    d.pad_mask[base + 1 + pos] = false;
                }
            }
            d.chord_labels.extend_from_slice(&ex.chord_labels);
            d.key_labels.push(ex.key_label);
        }
        d.n_snd = d.pitch.len();
        d.examples = examples;
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::{Instrument, NoteEvent};

    fn small_song() -> Song {
        Song {
            key_label: 0,
            n_frames: 8,
            notes: vec![
                NoteEvent {
                    start_frame: 0,
                    end_frame: 4,
                    pitch: 60,
                    velocity: 1.0,
                    instrument: Instrument::Bass,
                    track: 5,
                    pan: 0.0,
                },
                NoteEvent {
                    start_frame: 2,
                    end_frame: 3,
                    pitch: 67,
                    velocity: 0.5,
                    instrument: Instrument::Melody,
                    track: 3,
                    pan: 0.0,
                },
            ],
            chord_labels: vec![0; 8],
            is_music: None,
        }
    }

    #[test]
    fn hier_batch_flat_layout() {
        let data = HierBatchData::build(std::slice::from_ref(&small_song()));
        let stride = MAX_POLY + 1;
        // note 1 sounds frames 0..4 (4), note 2 sounds frame 2 (1) → 5 (note, frame) pairs.
        assert_eq!(data.batch, 1);
        assert_eq!(data.n_frames, 8);
        assert_eq!(data.n_snd, 5);
        assert_eq!(data.pad_mask.len(), 8 * stride);

        // Frame 0: CLS + 1 note (pos 0) valid; pos 1 padded.
        assert!(!data.pad_mask[0]);
        assert!(!data.pad_mask[1]);
        assert!(data.pad_mask[2]);
        // Frame 2: CLS + 2 notes (pos 0,1) valid; pos 2 padded.
        let base2 = 2 * stride;
        assert!(!data.pad_mask[base2]);
        assert!(!data.pad_mask[base2 + 1]);
        assert!(!data.pad_mask[base2 + 2]);
        assert!(data.pad_mask[base2 + 3]);
        // Frame 4: empty except CLS.
        let base4 = 4 * stride;
        assert!(!data.pad_mask[base4]);
        assert!(data.pad_mask[base4 + 1]);

        // Each note lands in a distinct grid slot.
        let mut sorted = data.slot_row.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), data.n_snd);

        // Channels carried through to the flat list.
        assert!(data.channel.contains(&5));
        assert!(data.channel.contains(&3));
    }

    /// A frame with more simultaneous notes than `MAX_POLY` drops the extras rather
    /// than overflowing its slot range.
    #[test]
    fn polyphony_is_capped_at_max_poly() {
        let notes = (0..MAX_POLY + 6)
            .map(|i| NoteEvent {
                start_frame: 0,
                end_frame: 2,
                pitch: 40 + i as u8,
                velocity: 1.0,
                instrument: Instrument::Harmony,
                track: 1,
                pan: 0.0,
            })
            .collect();
        let song = Song {
            key_label: 0,
            n_frames: 2,
            notes,
            chord_labels: vec![0; 2],
            is_music: None,
        };
        let data = HierBatchData::build(std::slice::from_ref(&song));
        // 2 frames x MAX_POLY kept, extras dropped.
        assert_eq!(data.n_snd, 2 * MAX_POLY);
        assert!(data.slot_row.iter().all(|&r| (r as usize) < 2 * MAX_POLY));
    }
}
