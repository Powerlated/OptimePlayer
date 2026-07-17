//! Shared argument parsing for the bins.
//!
//! The three model generations are selected at the command line rather than by
//! separate binaries — `train --backbone hier` instead of a `hier_train` copy of
//! `train`. Dispatch stays an explicit `match` at each call site (the loops are
//! generic over [`crate::backbone::Backbone`], so the concrete type must be named
//! somewhere), but the parsing lives here once.

use std::path::PathBuf;

/// Which model generation a bin should operate on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Kind {
    /// [`crate::m00_frame`] — hand-engineered feature grid.
    #[default]
    Frame,
    /// [`crate::m01_event`] — learned frame tokens, sum pooling.
    Event,
    /// [`crate::m02_hier`] — learned frame tokens, set-transformer pooling.
    Hier,
}

impl Kind {
    pub fn parse(s: &str) -> Option<Kind> {
        match s {
            "frame" | "m00" | "00" => Some(Kind::Frame),
            "event" | "m01" | "01" => Some(Kind::Event),
            "hier" | "m02" | "02" => Some(Kind::Hier),
            _ => None,
        }
    }

    /// Artifact subdirectory under the model root.
    pub fn dir(&self) -> &'static str {
        match self {
            Kind::Frame => "00-frame",
            Kind::Event => "01-event",
            Kind::Hier => "02-hier",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Kind::Frame => "frame",
            Kind::Event => "event",
            Kind::Hier => "hier",
        }
    }
}

/// Flags shared by the train / pretrain / eval bins, plus whatever positional
/// arguments the bin defines itself.
pub struct Args {
    /// `--backbone {frame,event,hier}` (default `frame`).
    pub kind: Kind,
    /// `--out-dir <path>` — model root the backbone's subdir hangs off
    /// (default `models`). Point it elsewhere to keep a throwaway run from
    /// overwriting real checkpoints.
    pub out_dir: PathBuf,
    /// `--pretrained <prefix>` — warm-start weights.
    pub pretrained: Option<PathBuf>,
    /// `--model <stem>` — artifact stem to load, under `<out_dir>/<DIR>/`. `None` = `model`, the
    /// synthetic fine-tune. Point it at `model_sft` to score the real-label stage: without this the
    /// eval bins can only ever load the checkpoint SFT *starts from*, so an SFT run cannot be
    /// measured.
    pub model: Option<String>,
    /// `--train-songs <id,...>` / `--val-songs <id,...>` — hand-picked song ids, overriding the
    /// deterministic hash holdout with a contrived split. See [`crate::annotations::Split::Songs`]:
    /// numbers from one are not comparable to a default run.
    pub train_songs: Option<Vec<u32>>,
    pub val_songs: Option<Vec<u32>>,
    /// Everything not consumed as a flag, in order.
    pub positional: Vec<String>,
}

/// `360,362` → `[360, 362]`.
fn song_ids(v: &str, flag: &str) -> Vec<u32> {
    v.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse()
                .unwrap_or_else(|e| panic!("{flag}: {s:?} is not a song id: {e:?}"))
        })
        .collect()
}

impl Args {
    /// Parse `std::env::args()`, skipping argv[0].
    pub fn parse() -> Args {
        Args::from_args(std::env::args().skip(1))
    }

    pub fn from_args<I: IntoIterator<Item = String>>(args: I) -> Args {
        let mut out = Args {
            kind: Kind::default(),
            out_dir: PathBuf::from("models"),
            pretrained: None,
            model: None,
            train_songs: None,
            val_songs: None,
            positional: Vec::new(),
        };
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--backbone" => {
                    let v = it.next().expect("--backbone needs a value");
                    out.kind = Kind::parse(&v)
                        .unwrap_or_else(|| panic!("unknown backbone {v:?} (frame|event|hier)"));
                }
                "--out-dir" => {
                    out.out_dir = PathBuf::from(it.next().expect("--out-dir needs a path"));
                }
                "--pretrained" => {
                    out.pretrained = Some(PathBuf::from(
                        it.next().expect("--pretrained needs a path prefix"),
                    ));
                }
                "--model" => {
                    out.model = Some(it.next().expect("--model needs an artifact stem"));
                }
                "--train-songs" => {
                    let v = it
                        .next()
                        .expect("--train-songs needs a comma-separated id list");
                    out.train_songs = Some(song_ids(&v, "--train-songs"));
                }
                "--val-songs" => {
                    let v = it
                        .next()
                        .expect("--val-songs needs a comma-separated id list");
                    out.val_songs = Some(song_ids(&v, "--val-songs"));
                }
                _ => out.positional.push(a),
            }
        }
        out
    }

    /// Positional `i` parsed as `T`, or `default`.
    pub fn positional_or<T: std::str::FromStr>(&self, i: usize, default: T) -> T
    where
        T::Err: std::fmt::Debug,
    {
        match self.positional.get(i) {
            Some(v) => v
                .parse()
                .unwrap_or_else(|e| panic!("arg {i} ({v:?}): {e:?}")),
            None => default,
        }
    }

    /// Default warm-start prefix for this backbone: `<out_dir>/<DIR>/pretrained`.
    pub fn default_pretrained(&self) -> PathBuf {
        self.out_dir.join(self.kind.dir()).join("pretrained")
    }

    /// Prefix of the checkpoint to load: `<out_dir>/<DIR>/<--model or "model">`.
    pub fn model_prefix(&self) -> PathBuf {
        self.out_dir
            .join(self.kind.dir())
            .join(self.model.as_deref().unwrap_or("model"))
    }

    /// The hand-picked split, if either song-list flag was given. `Split::Songs` on each side; a
    /// song named by neither is in neither half.
    pub fn explicit_split(&self) -> Option<(crate::annotations::Split, crate::annotations::Split)> {
        use crate::annotations::Split;
        if self.train_songs.is_none() && self.val_songs.is_none() {
            return None;
        }
        Some((
            Split::Songs(self.train_songs.clone().unwrap_or_default()),
            Split::Songs(self.val_songs.clone().unwrap_or_default()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Args {
        Args::from_args(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn defaults_to_frame_and_models_dir() {
        let a = args(&[]);
        assert_eq!(a.kind, Kind::Frame);
        assert_eq!(a.out_dir, PathBuf::from("models"));
        assert!(a.pretrained.is_none());
    }

    #[test]
    fn parses_flags_and_keeps_positional_order() {
        let a = args(&[
            "12",
            "--backbone",
            "hier",
            "64",
            "--out-dir",
            "/tmp/smoke",
            "--pretrained",
            "p/x",
            "3e-4",
        ]);
        assert_eq!(a.kind, Kind::Hier);
        assert_eq!(a.out_dir, PathBuf::from("/tmp/smoke"));
        assert_eq!(a.pretrained, Some(PathBuf::from("p/x")));
        assert_eq!(a.positional, vec!["12", "64", "3e-4"]);
        assert_eq!(a.positional_or(0, 0usize), 12);
        assert_eq!(a.positional_or(1, 0usize), 64);
        assert_eq!(a.positional_or(9, 7usize), 7);
    }

    #[test]
    fn parses_the_hand_picked_split_and_model_stem() {
        let a = args(&[
            "--train-songs",
            "360",
            "--val-songs",
            "362, 359",
            "--model",
            "model_sft",
            "--backbone",
            "hier",
        ]);
        assert_eq!(a.train_songs, Some(vec![360]));
        assert_eq!(a.val_songs, Some(vec![362, 359]));
        assert_eq!(a.model_prefix(), PathBuf::from("models/02-hier/model_sft"));
        let (train, val) = a.explicit_split().expect("both flags given");
        assert!(train.accepts("x", 360) && !train.accepts("x", 362));
        assert!(val.accepts("x", 362) && !val.accepts("x", 360));
    }

    /// Without the flags nothing changes: the deterministic hash holdout stays the default, and the
    /// eval bins keep loading `model`.
    #[test]
    fn no_song_flags_means_no_explicit_split() {
        let a = args(&["--backbone", "hier"]);
        assert!(a.explicit_split().is_none());
        assert_eq!(a.model_prefix(), PathBuf::from("models/02-hier/model"));
    }

    #[test]
    fn backbone_aliases_and_dirs() {
        assert_eq!(Kind::parse("m02"), Some(Kind::Hier));
        assert_eq!(Kind::parse("01"), Some(Kind::Event));
        assert_eq!(Kind::parse("nope"), None);
        assert_eq!(Kind::Frame.dir(), "00-frame");
        assert_eq!(Kind::Event.dir(), "01-event");
        assert_eq!(Kind::Hier.dir(), "02-hier");
    }
}
