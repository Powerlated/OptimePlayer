//! Time-throttled in-epoch progress logging.
//!
//! Each training epoch's inner batch loop otherwise runs silent until the epoch
//! summary. [`TrainProgress`] prints the running-average loss, batch progress, and
//! throughput at most once per `interval` (default ~1s), plus once when the epoch
//! finishes — so long epochs show live progress. The epoch summary is unaffected.
//!
//! The same samples feed [`crate::dashboard`], so the live web view and the console
//! never disagree: one throttle, one set of numbers, two sinks.

use crate::dashboard;
use std::time::{Duration, Instant};

/// Default cadence for [`TrainProgress::per_epoch`]: one line per second.
const DEFAULT_INTERVAL_SECS: f64 = 1.0;

/// Time-throttled in-epoch progress logger. One per epoch; call [`maybe_log`]
/// after every batch.
///
/// [`maybe_log`]: TrainProgress::maybe_log
pub struct TrainProgress {
    epoch: usize,
    start: Instant,
    last: Instant,
    interval: Duration,
}

impl TrainProgress {
    /// New logger for `epoch`, emitting at most every `interval_secs` (clamped to
    /// ≥50 ms).
    pub fn new(epoch: usize, interval_secs: f64) -> Self {
        let now = Instant::now();
        Self {
            epoch,
            start: now,
            last: now,
            interval: Duration::from_secs_f64(interval_secs.max(0.05)),
        }
    }

    /// Logger for `epoch` at the default ~1s cadence.
    pub fn per_epoch(epoch: usize) -> Self {
        Self::new(epoch, DEFAULT_INTERVAL_SECS)
    }

    /// Print the current running-average loss + progress if the interval has
    /// elapsed, or if the epoch just finished (`n_batches >= n_total`), and publish
    /// the same sample to [`crate::dashboard`]. `running` is the sum of per-batch
    /// losses so far this epoch; `n_batches`/`n_total` are batches done / batches in
    /// the epoch.
    pub fn maybe_log(&mut self, running: f64, n_batches: usize, n_total: usize) {
        let now = Instant::now();
        let done = n_total > 0 && n_batches >= n_total;
        if !done && now.duration_since(self.last) < self.interval {
            return;
        }
        self.last = now;
        let avg = if n_batches > 0 {
            running / n_batches as f64
        } else {
            0.0
        };
        let pct = if n_total > 0 {
            100.0 * n_batches as f64 / n_total as f64
        } else {
            0.0
        };
        let elapsed = self.start.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 {
            n_batches as f64 / elapsed
        } else {
            0.0
        };
        let epoch = self.epoch;
        println!(
            "  [ep {epoch} {pct:>5.1}%] batch {n_batches}/{n_total}  loss {avg:.4}  {rate:.1} batch/s  {elapsed:.0}s"
        );
        dashboard::record_batch(epoch, n_batches, n_total, avg, rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_logs_when_epoch_done() {
        let mut p = TrainProgress::new(1, 60.0); // interval long enough that time won't trigger
                                                 // Even with a long interval, finishing the epoch must log.
        p.maybe_log(3.0, 3, 3);
    }

    #[test]
    fn per_epoch_default_interval_is_one_second() {
        let p = TrainProgress::per_epoch(1);
        assert!((p.interval.as_secs_f64() - 1.0).abs() < 1e-9);
    }
}
