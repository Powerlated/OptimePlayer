//! Self-supervised pretraining on harvested real game songs → `<out-dir>/<NN>/pretrained`. Pretext
//! follows the backbone: `frame` = masked-frame MSE, `event`/`hier` = autoregressive next-frame
//! BCE. Reads `data/real_{train,val}.bin` (produced by `harvest`).

use super::opts::{Backbone, ModelOpts};
use clap::Args;
use optime_ml::backend::Back;
use optime_ml::data::load_songs;
use optime_ml::m01_event::{EventModel, EventModelConfig};
use optime_ml::m02_hier::{HierModel, HierModelConfig};
use optime_ml::pretrain::ar::{self, ArPretrainConfig};
use optime_ml::pretrain::masked::{self, PretrainConfig};

#[derive(Args, Debug)]
pub struct PretrainArgs {
    /// Epochs (default 20).
    pub epochs: Option<usize>,
    /// Batch size (default 32).
    pub batch_size: Option<usize>,
    /// Learning rate (default 3e-4).
    pub lr: Option<f64>,
    #[command(flatten)]
    pub opts: ModelOpts,
}

pub fn run(args: PretrainArgs) {
    let epochs = args.epochs.unwrap_or(20);
    let batch_size = args.batch_size.unwrap_or(32);
    let lr = args.lr.unwrap_or(3.0e-4);
    let out_dir = &args.opts.out_dir;

    // Songs are transposed on the fly during training (see the configs' `augment`).
    let train =
        load_songs("data/real_train.bin").expect("load data/real_train.bin (run `harvest` first)");
    let val = load_songs("data/real_val.bin").unwrap_or_default();
    println!(
        "loaded {} train / {} val real windows",
        train.len(),
        val.len()
    );

    match args.opts.backbone {
        // Generation 00: masked-frame reconstruction over the feature grid.
        Backbone::Frame => {
            let config = PretrainConfig::default()
                .with_epochs(epochs)
                .with_batch_size(batch_size)
                .with_lr(lr);
            masked::run(&config, &train, &val, out_dir);
        }
        // Learned-token generations: autoregressive next-frame prediction.
        Backbone::Event => {
            let config = ar_config(epochs, batch_size, lr);
            ar::run::<EventModel<Back>>(&config, &EventModelConfig::new(), &train, &val, out_dir);
        }
        Backbone::Hier => {
            let config = ar_config(epochs, batch_size, lr);
            ar::run::<HierModel<Back>>(&config, &HierModelConfig::new(), &train, &val, out_dir);
        }
    }
}

fn ar_config(epochs: usize, batch_size: usize, lr: f64) -> ArPretrainConfig {
    ArPretrainConfig::default()
        .with_epochs(epochs)
        .with_batch_size(batch_size)
        .with_lr(lr)
}
