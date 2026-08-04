use optime_core::{LoopAndTransitionOptions, PerDeviceSettings, SoundData, SynthController};

use crate::audio::ENGINE_SAMPLE_RATE_HZ;

const CHUNK_FRAMES: usize = 64;

const MAX_SECONDS: f64 = 480.0;

pub struct Bounce {
    pcm: Vec<(f32, f32)>,
    frame_at_step: Vec<u32>,
}

pub struct BounceJob {
    controller: SynthController,
    config: PerDeviceSettings,
    total_steps: u32,
    pcm: Vec<(f32, f32)>,
    frame_at_step: Vec<u32>,
    stalled: usize,
    last_step: u32,
    done: bool,
}

impl BounceJob {
    pub fn new(
        data: &dyn SoundData,
        song_id: u32,
        total_steps: u32,
        config: PerDeviceSettings,
    ) -> Option<BounceJob> {
        let mut controller = SynthController::new(ENGINE_SAMPLE_RATE_HZ, data, song_id)?;
        controller.set_loop_and_transition(LoopAndTransitionOptions::none());
        Some(BounceJob {
            controller,
            config,
            total_steps,
            pcm: Vec::new(),
            frame_at_step: Vec::with_capacity(total_steps as usize + 1),
            stalled: 0,
            last_step: u32::MAX,
            done: false,
        })
    }

    pub fn step(&mut self, budget_frames: usize) {
        if self.done {
            return;
        }
        let max_frames = (ENGINE_SAMPLE_RATE_HZ * MAX_SECONDS) as usize;
        let stall_limit = (ENGINE_SAMPLE_RATE_HZ as usize / CHUNK_FRAMES).max(1);
        let mut buf = [0.0f32; 2 * CHUNK_FRAMES];
        let target = (self.pcm.len() + budget_frames).min(max_frames);

        while self.pcm.len() < target {
            let step = self.controller.steps_elapsed();
            while (self.frame_at_step.len() as u32) <= step.min(self.total_steps) {
                self.frame_at_step.push(self.pcm.len() as u32);
            }
            if step >= self.total_steps {
                self.done = true;
                break;
            }
            self.stalled = if step == self.last_step {
                self.stalled + 1
            } else {
                0
            };
            if self.stalled >= stall_limit {
                self.done = true;
                break;
            }
            self.last_step = step;

            self.controller.fill(&mut buf, &self.config);
            self.pcm.extend(buf.chunks_exact(2).map(|f| (f[0], f[1])));
        }
        if self.pcm.len() >= max_frames {
            self.done = true;
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn progress(&self) -> f32 {
        if self.done {
            return 1.0;
        }
        (self.frame_at_step.len() as f32 / (self.total_steps.max(1) as f32)).clamp(0.0, 1.0)
    }

    pub fn finish(mut self) -> Bounce {
        let end = self.pcm.len() as u32;
        while (self.frame_at_step.len() as u32) <= self.total_steps {
            self.frame_at_step.push(end);
        }
        Bounce {
            pcm: self.pcm,
            frame_at_step: self.frame_at_step,
        }
    }
}

impl Bounce {
    pub fn frames(&self) -> usize {
        self.pcm.len()
    }

    #[cfg(test)]
    pub fn total_steps(&self) -> u32 {
        self.frame_at_step.len().saturating_sub(1) as u32
    }

    pub fn frame_of_step(&self, step: u32) -> usize {
        let i = (step as usize).min(self.frame_at_step.len().saturating_sub(1));
        self.frame_at_step.get(i).copied().unwrap_or(0) as usize
    }

    pub fn step_of_frame(&self, frame: usize) -> u32 {
        let f = frame as u32;
        match self.frame_at_step.binary_search(&f) {
            Ok(i) => {
                let mut i = i;
                while i + 1 < self.frame_at_step.len() && self.frame_at_step[i + 1] == f {
                    i += 1;
                }
                i as u32
            }
            Err(i) => i.saturating_sub(1) as u32,
        }
    }

    #[inline]
    pub fn frame(&self, i: usize) -> (f32, f32) {
        self.pcm.get(i).copied().unwrap_or((0.0, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake(frame_at_step: Vec<u32>, frames: usize) -> Bounce {
        Bounce {
            pcm: vec![(0.0, 0.0); frames],
            frame_at_step,
        }
    }

    #[test]
    fn step_and_frame_round_trip() {
        let b = fake(vec![0, 100, 250, 250, 400], 400);
        assert_eq!(b.total_steps(), 4);
        assert_eq!(b.frame_of_step(0), 0);
        assert_eq!(b.frame_of_step(2), 250);
        assert_eq!(b.step_of_frame(0), 0);
        assert_eq!(b.step_of_frame(99), 0);
        assert_eq!(b.step_of_frame(100), 1);
        assert_eq!(b.step_of_frame(249), 1);
        assert_eq!(b.step_of_frame(250), 3);
        assert_eq!(b.step_of_frame(399), 3);
    }

    #[test]
    fn lookups_clamp_out_of_range() {
        let b = fake(vec![0, 100, 200], 200);
        assert_eq!(b.frame_of_step(99), 200);
        assert_eq!(b.step_of_frame(10_000), 2);
        assert_eq!(b.frame(10_000), (0.0, 0.0));
    }

    #[test]
    fn renders_real_audio_with_a_usable_step_map() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../demos/super-mario-64-ds.sdat");
        let bytes = std::fs::read(path).expect("demo file should exist");
        let archives = optime_core::load_all(&bytes);
        let data = archives.first().expect("an archive");
        let song_id = *data.song_ids().first().expect("a playable song");

        const STEPS: u32 = 2_000;
        let config = PerDeviceSettings::neutral();
        let mut job = BounceJob::new(&**data, song_id, STEPS, config).expect("job");
        let mut guard = 0;
        while !job.is_done() && guard < 10_000 {
            job.step(4_096);
            guard += 1;
        }
        assert!(job.is_done(), "bounce should finish");
        let b = job.finish();

        assert!(b.frames() > 0, "should have rendered audio");
        assert_eq!(b.total_steps(), STEPS, "map covers the requested span");
        assert!(
            (0..b.frames()).any(|i| b.frame(i).0 != 0.0 || b.frame(i).1 != 0.0),
            "rendered buffer should not be silent"
        );
        for s in 1..=STEPS {
            assert!(
                b.frame_of_step(s) >= b.frame_of_step(s - 1),
                "frame_at_step must be non-decreasing at step {s}"
            );
        }
        for s in (0..STEPS).step_by(97) {
            let f = b.frame_of_step(s);
            if f >= b.frames() {
                continue;
            }
            let back = b.step_of_frame(f);
            assert!(
                back >= s,
                "step {s} → frame {f} → step {back} should not go backwards"
            );
            assert_eq!(
                b.frame_of_step(back),
                f,
                "steps sharing frame {f} must agree on it"
            );
        }
    }
}
