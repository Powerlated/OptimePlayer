//! Whole-song bounce: the annotation platform's audio source.
//!
//! Annotating harmony by ear needs random access — click a bar and hear it, loop two bars while
//! deciding, scrub backwards. The live path can do none of that: a `DevicePlayer` only ticks
//! forward, so **there is no seek in the engine at all**, and adding one would mean fast-forwarding
//! the sequencer (approximate, and audibly seamed once a fade is involved).
//!
//! So annotation renders the song to memory **once** instead. Every "seek" is then an index into
//! that buffer: sample-exact, instant, and seamless to loop — no fade, no re-decode, no sequencer
//! state to reconstruct.
//!
//! Only the *audio* needs bouncing. The note timeline is already whole-song: `SongOverview`
//! ([`optime_core::SongOverview`], built per song at load) holds every note. [`Bounce`] carries the
//! step→frame map that ties the two into one coordinate system, so the piano roll's step axis and
//! the audio buffer address the same music.
//!
//! Rendering is **incremental** ([`BounceJob`]) rather than threaded: annotation runs on the web
//! too, where there is no thread to spawn, and a multi-second block would freeze the tab. Slicing
//! the render across frames costs nothing on native, works identically on both targets, and gives a
//! progress readout for free.

use optime_core::{LoopAndTransitionOptions, PerDeviceSettings, SoundData, SynthController};

use crate::audio::ENGINE_SAMPLE_RATE_HZ;

/// Frames rendered per `fill` call. Sets how finely the step→frame map is sampled (the position is
/// read once per chunk), so it is a resolution knob, not a performance one: 64 frames is ~2 ms at
/// [`ENGINE_SAMPLE_RATE_HZ`], well under one sequencer step on every device.
const CHUNK_FRAMES: usize = 64;

/// Hard cap on a bounce, mirroring the offline renderer's own guard: a song that neither loops nor
/// ends can't eat memory forever. 480 s ≈ 126 MB of stereo `f32`.
const MAX_SECONDS: f64 = 480.0;

/// A song rendered to memory, plus the map tying it to the piano roll's step timeline.
pub struct Bounce {
    /// Stereo frames at [`ENGINE_SAMPLE_RATE_HZ`] — the same rate the live path renders at, so
    /// playback reuses the output `StreamResampler` unchanged and sounds identical to live.
    pcm: Vec<(f32, f32)>,
    /// `frame_at_step[s]` = first frame at sequencer step `s`; length `total_steps + 1`. Monotonic
    /// non-decreasing, so frame→step is a binary search ([`Self::step_of_frame`]).
    frame_at_step: Vec<u32>,
}

/// An in-progress [`Bounce`]: renders `song_id` over `[0, total_steps)` — the same span
/// `FsVisController::overview` covers, so the bounced audio and the roll's notes always agree.
///
/// Driven by [`Self::step`] a slice at a time from the UI loop. **Not threaded on purpose**: web has
/// no thread to spawn, and rendering minutes of audio in one go would freeze the tab. Slicing costs
/// nothing on native and makes both targets identical.
///
/// Uses [`LoopAndTransitionOptions::none`]: no loop-fade, no end-fade. A faded buffer would be wrong
/// here — annotation loops a bar over and over, and every repeat must sound the same.
pub struct BounceJob {
    controller: SynthController,
    config: PerDeviceSettings,
    total_steps: u32,
    pcm: Vec<(f32, f32)>,
    frame_at_step: Vec<u32>,
    /// Consecutive chunks the sequencer hasn't advanced through.
    stalled: usize,
    last_step: u32,
    done: bool,
}

impl BounceJob {
    /// Starts a bounce. `None` if the song can't be built.
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

    /// Renders up to `budget_frames` more frames. Call until [`Self::is_done`].
    pub fn step(&mut self, budget_frames: usize) {
        if self.done {
            return;
        }
        let max_frames = (ENGINE_SAMPLE_RATE_HZ * MAX_SECONDS) as usize;
        // A device that stops advancing (ended early, malformed sequence) would otherwise render to
        // the frame cap for nothing. One second of silence-in-place is decisive.
        let stall_limit = (ENGINE_SAMPLE_RATE_HZ as usize / CHUNK_FRAMES).max(1);
        let mut buf = [0.0f32; 2 * CHUNK_FRAMES];
        let target = (self.pcm.len() + budget_frames).min(max_frames);

        while self.pcm.len() < target {
            let step = self.controller.steps_elapsed();
            // Every step reached since the last chunk starts at the current frame.
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

    /// Rendered fraction, by sequencer step (`0..=1`) — steps are the axis the user sees.
    pub fn progress(&self) -> f32 {
        if self.done {
            return 1.0;
        }
        (self.frame_at_step.len() as f32 / (self.total_steps.max(1) as f32)).clamp(0.0, 1.0)
    }

    /// Finishes the bounce, padding the step map so every step in `[0, total_steps]` is addressable
    /// even if the render stopped early — lookups then clamp to the end of the buffer instead of
    /// going out of bounds.
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
    /// Total rendered frames.
    pub fn frames(&self) -> usize {
        self.pcm.len()
    }

    /// Length of the bounced span in sequencer steps.
    #[cfg(test)]
    pub fn total_steps(&self) -> u32 {
        self.frame_at_step.len().saturating_sub(1) as u32
    }

    /// The frame a sequencer step starts at, clamped to the buffer.
    pub fn frame_of_step(&self, step: u32) -> usize {
        let i = (step as usize).min(self.frame_at_step.len().saturating_sub(1));
        self.frame_at_step.get(i).copied().unwrap_or(0) as usize
    }

    /// The sequencer step a frame falls in — the inverse of [`Self::frame_of_step`], by binary
    /// search over the monotonic map. Drives the roll's playhead while a bounce is playing.
    pub fn step_of_frame(&self, frame: usize) -> u32 {
        let f = frame as u32;
        match self.frame_at_step.binary_search(&f) {
            // Landed on a boundary; several steps can share a frame, so take the last of them.
            Ok(i) => {
                let mut i = i;
                while i + 1 < self.frame_at_step.len() && self.frame_at_step[i + 1] == f {
                    i += 1;
                }
                i as u32
            }
            // Between marks: the step that started before this frame.
            Err(i) => i.saturating_sub(1) as u32,
        }
    }

    /// One stereo frame, or silence past the end.
    #[inline]
    pub fn frame(&self, i: usize) -> (f32, f32) {
        self.pcm.get(i).copied().unwrap_or((0.0, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a `Bounce` directly, bypassing rendering (which needs a real device).
    fn fake(frame_at_step: Vec<u32>, frames: usize) -> Bounce {
        Bounce {
            pcm: vec![(0.0, 0.0); frames],
            frame_at_step,
        }
    }

    #[test]
    fn step_and_frame_round_trip() {
        // Steps 0..=3 start at frames 0, 100, 250, 250 (a step with no audio of its own), end 400.
        let b = fake(vec![0, 100, 250, 250, 400], 400);
        assert_eq!(b.total_steps(), 4);
        assert_eq!(b.frame_of_step(0), 0);
        assert_eq!(b.frame_of_step(2), 250);
        // Inside a step maps back to that step.
        assert_eq!(b.step_of_frame(0), 0);
        assert_eq!(b.step_of_frame(99), 0);
        assert_eq!(b.step_of_frame(100), 1);
        assert_eq!(b.step_of_frame(249), 1);
        // Zero-length step 2 shares frame 250 with step 3: the *last* step at that frame wins, so
        // the playhead never appears to stall on a step that occupies no audio.
        assert_eq!(b.step_of_frame(250), 3);
        assert_eq!(b.step_of_frame(399), 3);
    }

    #[test]
    fn lookups_clamp_out_of_range() {
        let b = fake(vec![0, 100, 200], 200);
        assert_eq!(b.frame_of_step(99), 200); // past the end → end of buffer
        assert_eq!(b.step_of_frame(10_000), 2);
        assert_eq!(b.frame(10_000), (0.0, 0.0)); // silence, not a panic
    }

    /// End-to-end against a real demo: the incremental job must actually render audio, and the
    /// step↔frame map it builds must be usable — that map is the whole contract between the roll's
    /// timeline and the buffer, so a mistake in it silently mis-seeks every annotation.
    #[test]
    fn renders_real_audio_with_a_usable_step_map() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../demos/super-mario-64-ds.sdat");
        let bytes = std::fs::read(path).expect("demo file should exist");
        let archives = optime_core::load_all(&bytes);
        let data = archives.first().expect("an archive");
        let song_id = *data.song_ids().first().expect("a playable song");

        // A short span keeps the test quick; the map's properties don't depend on length.
        const STEPS: u32 = 2_000;
        let config = PerDeviceSettings::neutral();
        let mut job = BounceJob::new(&**data, song_id, STEPS, config).expect("job");
        // Drive it exactly as the UI does — in slices — so the slicing itself is under test.
        let mut guard = 0;
        while !job.is_done() && guard < 10_000 {
            job.step(4_096);
            guard += 1;
        }
        assert!(job.is_done(), "bounce should finish");
        let b = job.finish();

        assert!(b.frames() > 0, "should have rendered audio");
        assert_eq!(b.total_steps(), STEPS, "map covers the requested span");
        // Not silence — a bounce of zeros would look fine to every other assertion here.
        assert!(
            (0..b.frames()).any(|i| b.frame(i).0 != 0.0 || b.frame(i).1 != 0.0),
            "rendered buffer should not be silent"
        );
        // Monotonic: time only moves forward, so a binary search over it is valid.
        for s in 1..=STEPS {
            assert!(
                b.frame_of_step(s) >= b.frame_of_step(s - 1),
                "frame_at_step must be non-decreasing at step {s}"
            );
        }
        // Round-trip: seeking to a step and asking where we are must agree.
        for s in (0..STEPS).step_by(97) {
            let f = b.frame_of_step(s);
            if f >= b.frames() {
                continue; // past the rendered tail
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
