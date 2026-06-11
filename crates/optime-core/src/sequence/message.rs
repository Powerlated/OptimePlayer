//! [`Message`]s emitted by a [`SequenceTrack`](super::SequenceTrack) for the controller to act on.

/// The kind of a [`Message`] emitted by the sequence to the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// `param0` = MIDI note, `param1` = velocity, `param2` = duration (ticks).
    PlayNote,
    /// `param0` = bank, `param1` = program.
    InstrumentChange,
    /// The track jumped (used for loop detection).
    Jump,
    /// The track ended.
    TrackEnded,
    /// `param0` = volume (0..127).
    VolumeChange,
    /// `param0` = pan (0..128).
    PanChange,
    /// Pitch bend / bend-range changed; the controller reads the track state.
    PitchBend,
}

/// A control message produced by a [`SequenceTrack`](super::SequenceTrack) for the controller to
/// act on.
#[derive(Debug, Clone, Copy)]
pub struct Message {
    /// Whether this note originated from live keyboard input rather than the sequence.
    pub from_keyboard: bool,
    /// Which track emitted it.
    pub track_num: usize,
    /// The message kind.
    pub msg_type: MessageType,
    /// First parameter (meaning depends on `msg_type`).
    pub param0: i32,
    /// Second parameter.
    pub param1: i32,
    /// Third parameter.
    pub param2: i32,
    /// Tick the message was generated (filled in by consumers that need it).
    pub timestamp: u32,
}
