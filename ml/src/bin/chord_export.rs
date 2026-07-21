//! Offline chord/key pre-inference (feature `harvest`).
//!
//! Runs the trained key/chord model over every playable song in a sound archive
//! and writes a compact **bespoke binary** (`.ocd`) the app's piano roll consumes to
//! draw scrolling chord labels — `<roman numeral> (<absolute>)`. The model is too
//! heavy (and unavailable on the wasm build) to run live, so this bakes the result
//! once on PC. All music theory (roman spelling relative to the inferred key) lives
//! here; the app just maps segment indices to interned label strings.
//!
//! Usage:
//!   cargo run --release --features harvest --bin chord_export -- \
//!       <archive> <out.ocd> [--names <song_names.json>] [--model frame|event]
//!
//! - `<archive>` a `.gba`/`.gbaaudio`/`.nds`/`.sdat` (gzip is transparently handled).
//! - `<out.ocd>` output path.
//! - `--names` optional `song_names` JSON; restricts output to its curated ids
//!   (skips SFX/jingles). Omit to export every playable song.
//! - `--model` backbone: `frame` (default, reads `models/model`) or `event` (reads
//!   `models/event_model`). Both backbones share this pipeline and the `.ocd`
//!   format, so the rest of the tool is model-agnostic.
//!
//! Reads the trained model from `models/` (like `bin/infer`).
//!
//! ## `.ocd` binary format (little-endian)
//!
//! ```text
//! magic        "OCHD"            (4 bytes)
//! version      u8                (= 1)
//! song_count   u32
//! label_count  u32
//! labels       label_count × { len: u16, utf8: [u8; len] }   — the dictionary
//! songs        song_count × {
//!     song_id    u32
//!     end_step   u32             — final boundary (last segment's end)
//!     seg_count  u32
//!     segments   seg_count × { start_step: u32, label_idx: u16 }
//! }
//! ```
//!
//! Segments are contiguous (the model labels every frame), so only starts are
//! stored; ends are reconstructed from the next start (last from `end_step`).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use burn::module::AutodiffModule;
use optime_ml::backbone::{self, Backbone};
use optime_ml::backend::{Back, Inner, MlDevice};
use optime_ml::cli::{Args, Kind};
use optime_ml::harvest::harvest_song_full;
use optime_ml::infer::{merge_segments, predict, Prediction};
use optime_ml::m00_frame::FrameModel;
use optime_ml::m01_event::EventModel;
use optime_ml::m02_hier::HierModel;
use optime_ml::m03_kda::KdaModel;
use optime_ml::notes::{NoteEvent, FRAMES_PER_BEAT};
use optime_ml::theory::Key;

use optime_core::{load_all, SoundData};

/// Window length fed to the model — the full context it has (a bidirectional
/// encoder over this many frames, no cross-window memory). Equals the model's
/// `max_seq_len` and the training sequence length (128 frames = 32 beats).
const SEQ_LEN: usize = 128;

/// One song's reduced chord timeline: contiguous segments as `(start_step, label)`
/// plus the final boundary step.
struct SongOut {
    end_step: u32,
    segments: Vec<(u32, String)>,
}

/// Type-erased predict fn over one windowed note slice — wraps whichever backbone
/// was loaded, so [`export_song`] is model-agnostic.
type Predict = Box<dyn Fn(&[NoteEvent], usize) -> Prediction>;

/// Load `prefix` as backbone `M` and erase it behind a [`Predict`] closure.
fn boxed_predict<M>(prefix: &Path, device: MlDevice) -> Predict
where
    M: Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: Backbone<Inner, Batch = <M as Backbone<Back>>::Batch>,
{
    let model = backbone::load::<M, Back>(prefix, &device);
    Box::new(move |notes, nf| predict::<M>(&model, notes, nf, &device))
}

fn main() {
    let args = Args::parse();
    if args.positional.len() < 2 {
        eprintln!(
            "usage: chord_export <archive> <out.ocd> [--names <song_names.json>]              [--backbone frame|event|hier|kda] [--out-dir models]"
        );
        std::process::exit(1);
    }
    let archive_path = args.positional[0].as_str();
    let out_path = args.positional[1].as_str();
    let raw: Vec<String> = std::env::args().collect();
    let names_filter = flag(&raw, "--names").map(|p| load_curated_ids(Path::new(&p)));

    let device = MlDevice::default();

    // Load the requested backbone and wrap its predict fn in a single closure so the
    // rest of the pipeline (harvest → windowed inference → `.ocd` encode) is shared
    // and the fragile binary format lives in exactly one place.
    let prefix = args.out_dir.join(args.kind.dir()).join("model");
    if !prefix.with_extension("json").exists() {
        eprintln!(
            "{}.json not found — run `cargo run --release --bin train -- --backbone {}` first",
            prefix.display(),
            args.kind.name()
        );
        std::process::exit(1);
    }
    let predict: Predict = match args.kind {
        Kind::Frame => boxed_predict::<FrameModel<Back>>(&prefix, device),
        Kind::Event => boxed_predict::<EventModel<Back>>(&prefix, device),
        Kind::Hier => boxed_predict::<HierModel<Back>>(&prefix, device),
        Kind::Kda => boxed_predict::<KdaModel<Back>>(&prefix, device),
    };

    let bytes = read_maybe_gzip(Path::new(archive_path));
    let archives = load_all(&bytes);
    if archives.is_empty() {
        eprintln!("no sound archives found in {archive_path}");
        std::process::exit(1);
    }

    // Keyed + sorted by song id for a stable, diff-friendly output.
    let mut songs: BTreeMap<u32, SongOut> = BTreeMap::new();
    for data in &archives {
        for id in data.song_ids() {
            if let Some(filter) = &names_filter {
                if !filter.contains(&id) {
                    continue;
                }
            }
            if let Some(out) = export_song(&predict, data.as_ref(), id) {
                println!("  song {id}: {} chord segments", out.segments.len());
                songs.insert(id, out);
            }
        }
    }

    let encoded = encode(&songs);
    std::fs::write(out_path, &encoded).expect("write output");
    println!(
        "wrote {out_path} ({} songs, {} KB)",
        songs.len(),
        encoded.len() / 1024
    );
}

/// Harvest one song, infer chords over sliding windows, reduce to step-timed
/// segments with baked labels. Returns `None` if the song can't be started or is
/// empty. `predict` runs the chosen backbone (frame or event) over one window.
fn export_song<P: Fn(&[NoteEvent], usize) -> Prediction>(
    predict: P,
    data: &dyn SoundData,
    id: u32,
) -> Option<SongOut> {
    let (notes, steps_per_beat) = harvest_song_full(data, id)?;
    let total_frames = notes.iter().map(|n| n.end_frame).max().unwrap_or(0) as usize;
    if total_frames == 0 {
        return None;
    }

    // Infer window-by-window (rebasing frames), concatenating per-frame labels and
    // remembering the most confident window's key as the song key (roman is spelled
    // relative to it).
    let mut labels: Vec<usize> = Vec::with_capacity(total_frames);
    let mut best_key = Key::from_label(0);
    let mut best_conf = -1.0_f32;
    let n_windows = total_frames.div_ceil(SEQ_LEN);
    for w in 0..n_windows {
        let win_start = (w * SEQ_LEN) as u32;
        let win_end = win_start + SEQ_LEN as u32;
        let win_notes = clip_window(&notes, win_start, win_end);
        let pred = predict(&win_notes, SEQ_LEN);
        if pred.key_confidence > best_conf {
            best_conf = pred.key_confidence;
            best_key = pred.key;
        }
        let take = (total_frames - w * SEQ_LEN).min(SEQ_LEN);
        labels.extend_from_slice(&pred.chord_labels[..take]);
    }

    // Frame → sequencer step (inverse of `harvest::step_to_frame`), rounded to a
    // whole step (GBA/DS beat grids divide evenly, so this is exact in practice).
    let to_step = |frame: usize| -> u32 {
        if steps_per_beat > 0.0 {
            (frame as f64 * steps_per_beat / FRAMES_PER_BEAT as f64).round() as u32
        } else {
            frame as u32
        }
    };

    let segments = merge_segments(&labels)
        .into_iter()
        .map(|seg| {
            let label = match seg.chord {
                Some(c) => format!("{} ({})", c.roman(&best_key), c.name()),
                None => "N.C.".to_string(),
            };
            (to_step(seg.start_frame), label)
        })
        .collect();

    Some(SongOut {
        end_step: to_step(total_frames),
        segments,
    })
}

/// Serialize the song table into the `.ocd` byte layout (see the module docs).
fn encode(songs: &BTreeMap<u32, SongOut>) -> Vec<u8> {
    // Intern labels into a dictionary in first-seen order.
    let mut index: HashMap<&str, u16> = HashMap::new();
    let mut dict: Vec<&str> = Vec::new();
    for out in songs.values() {
        for (_, label) in &out.segments {
            index.entry(label).or_insert_with(|| {
                dict.push(label);
                (dict.len() - 1) as u16
            });
        }
    }
    assert!(dict.len() <= u16::MAX as usize, "too many distinct labels");

    let mut buf = Vec::new();
    buf.extend_from_slice(b"OCHD");
    buf.push(1);
    buf.extend_from_slice(&(songs.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(dict.len() as u32).to_le_bytes());
    for label in &dict {
        buf.extend_from_slice(&(label.len() as u16).to_le_bytes());
        buf.extend_from_slice(label.as_bytes());
    }
    for (&id, out) in songs {
        buf.extend_from_slice(&id.to_le_bytes());
        buf.extend_from_slice(&out.end_step.to_le_bytes());
        buf.extend_from_slice(&(out.segments.len() as u32).to_le_bytes());
        for (start, label) in &out.segments {
            buf.extend_from_slice(&start.to_le_bytes());
            buf.extend_from_slice(&index[label.as_str()].to_le_bytes());
        }
    }
    buf
}

/// Notes overlapping `[win_start, win_end)`, clipped and rebased to the window.
fn clip_window(notes: &[NoteEvent], win_start: u32, win_end: u32) -> Vec<NoteEvent> {
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

/// The set of curated song ids from a `song_names` JSON (`[{"songId":N,...}]`).
fn load_curated_ids(path: &Path) -> std::collections::HashSet<u32> {
    #[derive(serde::Deserialize)]
    struct Entry {
        #[serde(rename = "songId")]
        song_id: u32,
    }
    let text = std::fs::read_to_string(path).expect("read names json");
    let entries: Vec<Entry> = serde_json::from_str(&text).expect("parse names json");
    entries.into_iter().map(|e| e.song_id).collect()
}

/// Read a file, transparently gunzipping gzip-magic input (like the app does).
fn read_maybe_gzip(path: &Path) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read archive");
    if raw.starts_with(&[0x1f, 0x8b]) {
        use std::io::Read;
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&raw[..])
            .read_to_end(&mut out)
            .expect("gunzip archive");
        out
    } else {
        raw
    }
}

/// Value of a `--flag <value>` argument, if present.
fn flag(args: &[String], name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).cloned()
}
