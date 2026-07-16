//! In-process web dashboard for a training run.
//!
//! Every training driver ([`crate::train`], [`crate::pretrain`]) publishes the same
//! metrics it already prints — per-batch running loss, per-epoch train/val loss — to
//! a process-wide [`Hub`], and serves them over HTTP from a background thread. Point
//! a browser at the printed URL to watch a run live.
//!
//! Zero new crates: a GET-only server on [`std::net::TcpListener`], JSON via the
//! `serde_json` already in the tree, and a Vue single-page app (`ml/dashboard/`)
//! baked into the binary with [`include_str!`] — so a run serves its dashboard with
//! no build step, no working-directory assumptions, and no network fetch.
//!
//! Environment:
//! * `ML_DASHBOARD=0` — don't serve (the run is otherwise unchanged).
//! * `ML_DASHBOARD_ADDR` — bind address, default `0.0.0.0:7878` (all interfaces, so
//!   a run on a headless/remote box is reachable over a LAN or Tailscale address).
//! * `ML_DASHBOARD_HOLD=1` — keep serving after the run finishes instead of exiting,
//!   so the final curves survive the last epoch. Off by default: bins must still
//!   exit on their own for scripted runs.
//!
//! Recording is unconditional and bounded ([`MAX_BATCH_POINTS`]); only the listener
//! is optional. Failure to bind is a warning, never fatal — a dashboard must not be
//! able to kill a training run.

use crate::notes::{Song, FRAMES_PER_BEAT};
use serde::Serialize;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Default bind address: every interface, so remote viewing works out of the box.
const DEFAULT_ADDR: &str = "0.0.0.0:7878";

/// Cap on retained per-batch points. `TrainProgress` emits ~1/s, so this is ~1h of
/// wall time; past it the oldest points drop. Bounds memory on multi-day runs.
const MAX_BATCH_POINTS: usize = 4000;

const INDEX_HTML: &str = include_str!("../dashboard/index.html");
const APP_JS: &str = include_str!("../dashboard/app.js");
const VUE_JS: &str = include_str!("../dashboard/vendor/vue.global.prod.js");

/// Reference tempo for every wall-clock figure the dashboard reports.
///
/// The model's time axis is *beats* ([`FRAMES_PER_BEAT`] frames each), not seconds —
/// harvested songs are resampled onto that grid and their real tempo is discarded.
/// So "seconds" and "hours" here are only meaningful against a stated tempo; this is
/// it. A 60 bpm song's window really is twice as long in wall time.
pub const REFERENCE_BPM: f64 = 120.0;

/// How much context one window gives the model, in the three units worth knowing.
#[derive(Serialize, Clone, Copy, Debug)]
pub struct ContextWindow {
    /// Frame tokens the trunk attends over — the sequence length.
    pub tokens: usize,
    pub beats: f64,
    /// Wall time at [`REFERENCE_BPM`].
    pub seconds: f64,
}

impl ContextWindow {
    pub fn from_frames(n_frames: usize) -> Self {
        let beats = n_frames as f64 / FRAMES_PER_BEAT as f64;
        Self {
            tokens: n_frames,
            beats,
            seconds: beats / (REFERENCE_BPM / 60.0),
        }
    }
}

/// Size of the dataset, in windows and in music.
#[derive(Serialize, Clone, Copy, Debug, Default)]
pub struct DataStats {
    pub train_windows: usize,
    pub val_windows: usize,
    /// Mean notes per training window — generation 01's φ cost scales with it.
    pub notes_per_window: f64,
    /// Beats of music across the training split, exactly (no tempo assumed).
    pub train_beats: f64,
    /// Those beats as hours at [`REFERENCE_BPM`].
    pub train_hours: f64,
    /// Distinct transpositions augmentation can reach (1 when augmentation is off).
    pub transpositions: usize,
    /// `train_hours * transpositions` — the material the model can be shown, **not**
    /// hours of distinct recorded music. Every transposition is the same performance
    /// shifted, so this inflates variety, not information.
    pub augmented_hours: f64,
}

impl DataStats {
    pub fn measure(train: &[Song], val: &[Song], transpositions: usize) -> Self {
        let frames: u64 = train.iter().map(|s| s.n_frames as u64).sum();
        let notes: u64 = train.iter().map(|s| s.notes.len() as u64).sum();
        let beats = frames as f64 / FRAMES_PER_BEAT as f64;
        let hours = beats / REFERENCE_BPM / 60.0;
        Self {
            train_windows: train.len(),
            val_windows: val.len(),
            notes_per_window: notes as f64 / train.len().max(1) as f64,
            train_beats: beats,
            train_hours: hours,
            transpositions,
            augmented_hours: hours * transpositions as f64,
        }
    }
}

/// What this run is, fixed for its lifetime. Set once by the driver via [`start`].
#[derive(Serialize, Clone, Debug)]
pub struct RunMeta {
    /// Pretext or objective, e.g. `"AR pretrain"` / `"supervised fine-tune"`.
    pub stage: String,
    /// Backbone name (`Backbone::NAME`).
    pub backbone: String,
    /// Compute backend, e.g. `"wgpu"` / `"ndarray (8-way DP)"`.
    pub backend: String,
    /// Needed for progress + ETA. Other hyperparameters live in the config blobs.
    pub epochs: usize,
    pub context: ContextWindow,
    pub data: DataStats,
    /// Trainable parameters (`Module::num_params`).
    pub params: usize,
    /// Matmul-only estimate for one window's forward pass — see [`crate::flops`].
    pub flops_per_window: u64,
    /// Architecture hyperparameters, serialized from the backbone's own `Config` so
    /// the dashboard never drifts from the model: new field in, new row out.
    pub model_config: serde_json::Value,
    /// Optimisation hyperparameters, serialized from the driver's own `Config`.
    pub train_config: serde_json::Value,
}

/// One completed epoch. The two stages report different held-out metrics — a
/// pretext has a val loss, the supervised fine-tune has accuracies — so both sets
/// are optional rather than forced into one number the stage doesn't have.
/// `t` is stamped by [`record_epoch`].
#[derive(Serialize, Clone, Copy, Debug, Default)]
pub struct EpochPoint {
    pub epoch: usize,
    /// Seconds since the run started, at the end of this epoch.
    pub t: f64,
    pub train_loss: f64,
    /// Wall time this epoch took.
    pub secs: f64,
    /// Held-out pretext loss (masked recon / AR). `None` for the fine-tune.
    pub val_loss: Option<f64>,
    /// Fine-tune only: fraction in 0..1.
    pub key_acc: Option<f64>,
    pub chord_acc: Option<f64>,
    /// Fine-tune only: predicted chord transitions per sequence (flicker proxy).
    pub changes: Option<f64>,
}

impl EpochPoint {
    /// An epoch of a self-supervised pretext: train + held-out loss.
    pub fn pretext(epoch: usize, train_loss: f64, val_loss: f64, secs: f64) -> Self {
        Self {
            epoch,
            train_loss,
            secs,
            val_loss: Some(val_loss),
            ..Default::default()
        }
    }

    /// An epoch of the supervised fine-tune: train loss + held-out accuracies.
    pub fn supervised(
        epoch: usize,
        train_loss: f64,
        key_acc: f64,
        chord_acc: f64,
        changes: f64,
        secs: f64,
    ) -> Self {
        Self {
            epoch,
            train_loss,
            secs,
            key_acc: Some(key_acc),
            chord_acc: Some(chord_acc),
            changes: Some(changes),
            ..Default::default()
        }
    }
}

/// One in-epoch progress sample — the same numbers [`crate::progress`] prints.
#[derive(Serialize, Clone, Copy, Debug)]
pub struct BatchPoint {
    /// Seconds since the run started.
    pub t: f64,
    pub epoch: usize,
    pub batch: usize,
    pub of: usize,
    /// Running mean loss over this epoch so far.
    pub loss: f64,
    pub rate: f64,
}

/// The whole served state. Serialized verbatim as `/api/state`.
#[derive(Serialize, Clone, Debug, Default)]
struct State {
    meta: Option<RunMeta>,
    epochs: Vec<EpochPoint>,
    batches: VecDeque<BatchPoint>,
    finished: bool,
    /// Seconds since the run started, stamped at serialization time.
    elapsed: f64,
    /// Where the trained artifact landed, once saved.
    saved_to: Option<String>,
}

/// Process-wide metrics sink. One training run per process, so one hub.
struct Hub {
    start: Instant,
    state: Mutex<State>,
}

static HUB: OnceLock<Hub> = OnceLock::new();

fn hub() -> &'static Hub {
    HUB.get_or_init(|| Hub {
        start: Instant::now(),
        state: Mutex::new(State::default()),
    })
}

/// Readable backend label from a Rust type name: module paths dropped, generic
/// structure kept — `burn_autodiff::backend::Autodiff<burn_wgpu::Wgpu<f32, u32>>`
/// becomes `Autodiff<Wgpu<f32, u32>>`. Lets a driver label itself from
/// [`std::any::type_name`] without every caller passing a string.
pub fn backend_label(type_name: &str) -> String {
    fn flush(token: &mut String, out: &mut String) {
        if !token.is_empty() {
            out.push_str(token.rsplit("::").next().unwrap_or(token));
            token.clear();
        }
    }
    let mut out = String::new();
    let mut token = String::new();
    for c in type_name.chars() {
        if c.is_alphanumeric() || c == '_' || c == ':' {
            token.push(c);
        } else {
            flush(&mut token, &mut out);
            out.push(c);
        }
    }
    flush(&mut token, &mut out);
    out
}

/// Publish `meta` and start serving, unless `ML_DASHBOARD=0`. Returns the bound
/// address (already printed) or `None` if disabled or unbindable.
pub fn start(meta: RunMeta) -> Option<SocketAddr> {
    let h = hub();
    h.state.lock().unwrap().meta = Some(meta);

    if std::env::var("ML_DASHBOARD").as_deref() == Ok("0") {
        return None;
    }
    let addr = std::env::var("ML_DASHBOARD_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.to_string());
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("dashboard: could not bind {addr} ({e}) — training continues without it");
            return None;
        }
    };
    let bound = listener.local_addr().ok()?;
    println!("dashboard: http://{bound}  (ML_DASHBOARD=0 disables)");

    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            std::thread::spawn(move || handle(stream));
        }
    });
    Some(bound)
}

/// Record one in-epoch progress sample. Called from [`crate::progress`].
pub fn record_batch(epoch: usize, batch: usize, of: usize, loss: f64, rate: f64) {
    let h = hub();
    let t = h.start.elapsed().as_secs_f64();
    let mut s = h.state.lock().unwrap();
    s.batches.push_back(BatchPoint {
        t,
        epoch,
        batch,
        of,
        loss,
        rate,
    });
    while s.batches.len() > MAX_BATCH_POINTS {
        s.batches.pop_front();
    }
}

/// Record one completed epoch, stamping [`EpochPoint::t`] so the live chart can
/// place it on the same wall-time axis as the batch stream.
pub fn record_epoch(point: EpochPoint) {
    let h = hub();
    let t = h.start.elapsed().as_secs_f64();
    h.state
        .lock()
        .unwrap()
        .epochs
        .push(EpochPoint { t, ..point });
}

/// Mark the run finished and note where weights landed. If `ML_DASHBOARD_HOLD=1`
/// and a listener is up, park the calling thread so the dashboard outlives the run
/// (Ctrl-C to exit); otherwise return and let the bin exit as usual.
pub fn finish(saved_to: &std::path::Path) {
    {
        let mut s = hub().state.lock().unwrap();
        s.finished = true;
        s.saved_to = Some(saved_to.display().to_string());
    }
    let serving = std::env::var("ML_DASHBOARD").as_deref() != Ok("0");
    if serving && std::env::var("ML_DASHBOARD_HOLD").as_deref() == Ok("1") {
        println!("dashboard: run finished; still serving (ML_DASHBOARD_HOLD=1). Ctrl-C to exit.");
        loop {
            std::thread::park();
        }
    }
}

/// `/api/state` body: the snapshot with `elapsed` stamped now.
fn snapshot_json() -> String {
    let h = hub();
    let mut s = h.state.lock().unwrap().clone();
    s.elapsed = h.start.elapsed().as_secs_f64();
    serde_json::to_string(&s).unwrap_or_else(|_| "{}".to_string())
}

/// Serve one connection: parse the request line, ignore the rest, answer, close.
/// GET-only and read-only — the dashboard cannot influence the run.
fn handle(mut stream: TcpStream) {
    let peek = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut reader = BufReader::new(peek);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    // Drain headers so the client sees a clean response rather than a reset.
    let mut header = String::new();
    loop {
        header.clear();
        match reader.read_line(&mut header) {
            Ok(0) | Err(_) => break,
            Ok(_) if header.trim().is_empty() => break,
            Ok(_) => {}
        }
    }

    let path = request_line.split_whitespace().nth(1).unwrap_or("/");
    let (status, content_type, body) = match path {
        "/" | "/index.html" => ("200 OK", "text/html; charset=utf-8", INDEX_HTML.to_string()),
        "/app.js" => (
            "200 OK",
            "application/javascript; charset=utf-8",
            APP_JS.to_string(),
        ),
        "/vendor/vue.global.prod.js" => (
            "200 OK",
            "application/javascript; charset=utf-8",
            VUE_JS.to_string(),
        ),
        "/api/state" => ("200 OK", "application/json; charset=utf-8", snapshot_json()),
        _ => ("404 Not Found", "text/plain; charset=utf-8", String::new()),
    };

    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_points_are_bounded() {
        for i in 0..MAX_BATCH_POINTS + 50 {
            record_batch(1, i, MAX_BATCH_POINTS + 50, 1.0, 1.0);
        }
        assert_eq!(hub().state.lock().unwrap().batches.len(), MAX_BATCH_POINTS);
    }

    #[test]
    fn snapshot_is_valid_json_with_no_run() {
        let v: serde_json::Value = serde_json::from_str(&snapshot_json()).unwrap();
        assert!(v.get("epochs").unwrap().is_array());
    }

    #[test]
    fn backend_label_drops_paths_and_keeps_generics() {
        assert_eq!(
            backend_label("burn_autodiff::backend::Autodiff<burn_wgpu::Wgpu<f32, u32>>"),
            "Autodiff<Wgpu<f32, u32>>"
        );
        assert_eq!(backend_label("burn::backend::NdArray<f32>"), "NdArray<f32>");
    }
}
