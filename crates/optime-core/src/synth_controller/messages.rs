//! The message set every device speaks to the synth, and the feedback returned to it each tick.

use std::sync::Arc;

use crate::waveform::Waveform;

pub type VoiceId = u64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoicePitch {
    Midi { note: f64, sample_pitch_hz: f64 },
    DataRateHz(f64),
}

#[derive(Debug, Clone)]
pub enum SynthEvent {
    NoteStarted {
        track: usize,
        voice: VoiceId,
        key: u8,
        waveform: Arc<Waveform>,
        pitch: VoicePitch,
        volume: f64,
        duration_ticks: Option<u32>,
    },
    VoiceVolume {
        track: usize,
        voice: VoiceId,
        volume: f64,
    },
    VoicePitch {
        track: usize,
        voice: VoiceId,
        pitch: VoicePitch,
    },
    VoiceDetune {
        track: usize,
        voice: VoiceId,
        semitones: f64,
    },
    VoiceStopped {
        track: usize,
        voice: VoiceId,
    },
    NoteReleased {
        track: usize,
        key: u8,
    },
    TrackPan {
        track: usize,
        pan_vol_l: f64,
        pan_vol_r: f64,
    },
    TrackDetune {
        track: usize,
        semitones: f64,
    },
    ReverbAmount {
        amount: u8,
    },
    Looped,
    Ended,
}

#[derive(Debug, Default)]
pub struct TickFeedback {
    pub ended_voices: Vec<(usize, VoiceId)>,
}

impl TickFeedback {
    pub fn is_ended(&self, track: usize, voice: VoiceId) -> bool {
        self.ended_voices.contains(&(track, voice))
    }
}
