//! Shared CLI options: backbone selection + the artifact-path flags every train/eval subcommand
//! carries. Replaces the old hand-rolled `cli::Args`; the `Backbone`→path mapping and the
//! hand-picked-split logic are the same, just fed by clap instead of a manual arg scan.

use clap::{Args, ValueEnum};
use optime_ml::annotations::Split;
use std::path::PathBuf;

/// Which model generation a subcommand operates on. Aliases match the old `--backbone` spellings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum Backbone {
    /// `m00_frame` — hand-engineered feature grid.
    #[default]
    #[value(name = "frame", alias = "m00", alias = "00")]
    Frame,
    /// `m01_event` — learned frame tokens, sum pooling.
    #[value(name = "event", alias = "m01", alias = "01")]
    Event,
    /// `m02_hier` — learned frame tokens, set-transformer pooling.
    #[value(name = "hier", alias = "m02", alias = "02")]
    Hier,
}

impl Backbone {
    /// Artifact subdirectory under the model root.
    pub fn dir(&self) -> &'static str {
        match self {
            Backbone::Frame => "00-frame",
            Backbone::Event => "01-event",
            Backbone::Hier => "02-hier",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Backbone::Frame => "frame",
            Backbone::Event => "event",
            Backbone::Hier => "hier",
        }
    }
}

/// Flags shared by the train / pretrain / eval subcommands.
#[derive(Args, Debug, Clone)]
pub struct ModelOpts {
    /// Model generation to operate on.
    #[arg(long, value_enum, default_value_t = Backbone::Frame)]
    pub backbone: Backbone,

    /// Model root the backbone's subdir hangs off. Point it elsewhere to keep a throwaway run
    /// from overwriting real checkpoints.
    #[arg(long, default_value = "models")]
    pub out_dir: PathBuf,

    /// Warm-start weights prefix (an SSL/synthetic checkpoint).
    #[arg(long)]
    pub pretrained: Option<PathBuf>,

    /// Artifact stem to load under `<out_dir>/<DIR>/`. Default `model` (the synthetic fine-tune);
    /// point it at `model_sft` to score the real-label stage.
    #[arg(long)]
    pub model: Option<String>,

    /// Hand-picked training song ids (comma-separated), overriding the deterministic hash holdout
    /// with a contrived split. Not comparable to a default run.
    #[arg(long, value_delimiter = ',')]
    pub train_songs: Option<Vec<u32>>,

    /// Hand-picked validation song ids (comma-separated). See `--train-songs`.
    #[arg(long, value_delimiter = ',')]
    pub val_songs: Option<Vec<u32>>,
}

// `model_prefix`/`explicit_split` are exercised by the harvest-gated subcommands (and the tests);
// a plain default build compiles neither of those callers, so allow them to look unused there.
#[cfg_attr(not(feature = "harvest"), allow(dead_code))]
impl ModelOpts {
    /// Prefix of the checkpoint to load: `<out_dir>/<DIR>/<--model or "model">`.
    pub fn model_prefix(&self) -> PathBuf {
        self.out_dir
            .join(self.backbone.dir())
            .join(self.model.as_deref().unwrap_or("model"))
    }

    /// The hand-picked split, if either song-list flag was given. `Split::Songs` on each side; a
    /// song named by neither is in neither half.
    pub fn explicit_split(&self) -> Option<(Split, Split)> {
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
    use clap::Parser;

    /// A tiny harness command so `ModelOpts` can be exercised through the real clap parser.
    #[derive(Parser)]
    struct Harness {
        #[command(flatten)]
        opts: ModelOpts,
    }

    fn parse(argv: &[&str]) -> ModelOpts {
        let mut full = vec!["harness"];
        full.extend_from_slice(argv);
        Harness::parse_from(full).opts
    }

    #[test]
    fn defaults_to_frame_and_models_dir() {
        let o = parse(&[]);
        assert_eq!(o.backbone, Backbone::Frame);
        assert_eq!(o.out_dir, PathBuf::from("models"));
        assert!(o.pretrained.is_none());
    }

    #[test]
    fn backbone_aliases_and_dirs() {
        assert_eq!(parse(&["--backbone", "m02"]).backbone, Backbone::Hier);
        assert_eq!(parse(&["--backbone", "01"]).backbone, Backbone::Event);
        assert_eq!(Backbone::Frame.dir(), "00-frame");
        assert_eq!(Backbone::Event.dir(), "01-event");
        assert_eq!(Backbone::Hier.dir(), "02-hier");
    }

    #[test]
    fn parses_the_hand_picked_split_and_model_stem() {
        let o = parse(&[
            "--train-songs",
            "360",
            "--val-songs",
            "362,359",
            "--model",
            "model_sft",
            "--backbone",
            "hier",
        ]);
        assert_eq!(o.train_songs, Some(vec![360]));
        assert_eq!(o.val_songs, Some(vec![362, 359]));
        assert_eq!(o.model_prefix(), PathBuf::from("models/02-hier/model_sft"));
        let (train, val) = o.explicit_split().expect("both flags given");
        assert!(train.accepts("x", 360) && !train.accepts("x", 362));
        assert!(val.accepts("x", 362) && !val.accepts("x", 360));
    }

    #[test]
    fn no_song_flags_means_no_explicit_split() {
        let o = parse(&["--backbone", "hier"]);
        assert!(o.explicit_split().is_none());
        assert_eq!(o.model_prefix(), PathBuf::from("models/02-hier/model"));
    }
}
