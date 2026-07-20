//! [`SequenceTrack`]: the per-track interpreter state (registers, stack, LFO/portamento/flags).

/// Per-track interpreter state. Contains no back-reference to its [`Sequence`](super::Sequence);
/// the sequence drives each track by index, which keeps the borrow checker happy and mirrors the
/// hardware's flat array of channels.
#[derive(Debug, Clone)]
pub struct SequenceTrack {
    /// Whether the track is currently executing.
    pub active: bool,
    /// Track tempo (only track 0's BPM drives the sequence clock).
    pub bpm: u32,
    /// Program counter (offset within the SSEQ data region).
    pub pc: u32,
    /// Pan (0..128).
    pub pan: i32,
    /// Mono/poly flag.
    pub mono: bool,
    /// Channel volume (0..127).
    pub volume: i32,
    /// Master volume (0..127).
    pub master_volume: i32,
    /// Track priority.
    pub priority: i32,
    /// Selected program (instrument).
    pub program: usize,
    /// Selected bank.
    pub bank: usize,
    /// LFO waveform type.
    pub lfo_type: i32,
    /// LFO depth.
    pub lfo_depth: i32,
    /// LFO range multiplier.
    pub lfo_range: i32,
    /// LFO speed.
    pub lfo_speed: i32,
    /// LFO delay (in ticks).
    pub lfo_delay: i32,
    /// Raw pitch-bend value.
    pub pitch_bend: i32,
    /// Pitch-bend range in semitones.
    pub pitch_bend_range: i32,
    /// Expression controller.
    pub expression: i32,
    /// Portamento enable.
    pub portamento_enable: i32,
    /// Portamento time.
    pub portamento_time: i32,
    /// Portamento source key (pokediamond `portamentoKey`, default 60).
    pub portamento_key: i32,
    /// Note transposition in semitones (`0xC3`).
    pub transpose: i32,
    /// Sweep-pitch amount (`0xE3`).
    pub sweep_pitch: i32,
    /// "Note wait" flag (pokediamond `flags.noteWait`, default true): when set, a note advances
    /// the track clock by its duration (the DS default), rather than relying on explicit rests.
    pub note_wait: bool,
    /// Whether the track is muted (`0xC8` tie / `0xD7` mute paths).
    pub muted: bool,
    /// Tie flag (`0xC8`).
    pub tie: bool,
    /// Set after a zero-duration note in note-wait mode: the track stalls until its channels
    /// finish (pokediamond `flags.noteFinishWait`).
    pub note_finish_wait: bool,
    /// Conditional-execution flag set by the compare commands (`0xB8`–`0xBD`); gates the next
    /// command after an `0xA2` prefix (pokediamond `flags.cmp`, default true).
    pub cmp: bool,
    /// Remaining ticks to rest before executing again.
    pub resting_for: u32,
    /// Call/return stack.
    pub stack: [u32; 64],
    /// Per-frame loop counters paralleling [`Self::stack`] (`0xD4`/`0xFC`).
    pub loop_count: [u8; 64],
    /// Stack pointer.
    pub sp: usize,
    /// Sequence-overridden ADSR rates (currently informational).
    pub attack_rate: i32,
    /// See [`Self::attack_rate`].
    pub decay_rate: i32,
    /// See [`Self::attack_rate`].
    pub sustain_rate: i32,
    /// See [`Self::attack_rate`].
    pub release_rate: i32,
}

impl Default for SequenceTrack {
    fn default() -> Self {
        Self {
            active: false,
            bpm: 0,
            pc: 0,
            // Defaults mirror pokediamond's `TrackInit`.
            pan: 64, // 0..128 representation; 64 == centre (pokediamond's signed 0)
            mono: false,
            volume: 127,
            master_volume: 127,
            priority: 64,
            program: 0,
            bank: 0,
            lfo_type: 0,
            lfo_depth: 0,
            lfo_range: 1,
            lfo_speed: 16,
            lfo_delay: 0,
            pitch_bend: 0,
            pitch_bend_range: 2,
            expression: 127,
            portamento_enable: 0,
            portamento_time: 0,
            portamento_key: 60,
            transpose: 0,
            sweep_pitch: 0,
            note_wait: true,
            muted: false,
            tie: false,
            note_finish_wait: false,
            cmp: true,
            resting_for: 0,
            stack: [0; 64],
            loop_count: [0; 64],
            sp: 0,
            attack_rate: 0,
            decay_rate: 0,
            sustain_rate: 0,
            release_rate: 0,
        }
    }
}
