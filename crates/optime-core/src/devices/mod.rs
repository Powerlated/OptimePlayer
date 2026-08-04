pub mod dse;
pub mod gba;
pub mod nds;

pub use crate::synth_controller::messages::{SynthEvent, TickFeedback, VoiceId, VoicePitch};

use crate::PerDeviceSettings;

#[derive(Debug, Clone)]
pub struct WaveformDcStat {
    pub label: String,
    pub dc_shift: f32,
    pub length: usize,
    pub sample_rate: f64,
}

pub trait SoundData: Send + Sync {
    fn song_ids(&self) -> Vec<u32>;

    fn make_player(&self, id: u32) -> Option<Box<dyn DevicePlayer>>;

    fn song_name(&self, _id: u32) -> Option<String> {
        None
    }

    fn waveform_dc_stats(&self, _id: u32) -> Vec<WaveformDcStat> {
        Vec::new()
    }

    fn as_any(&self) -> &dyn std::any::Any;

    fn song_length_seconds(&self, id: u32) -> Option<f64> {
        let mut player = self.make_player(id)?;
        let tick_rate = player.tick_rate();
        let config = PerDeviceSettings::neutral();
        let mut feedback = TickFeedback::default();
        let mut events = Vec::new();
        let max_ticks = (tick_rate * 15.0 * 60.0) as u64;
        let mut ticks: u64 = 0;
        let mut loops = 0u32;
        let mut end_ticks = None;
        while ticks < max_ticks {
            events.clear();
            player.tick(&mut feedback, &config, &mut events);
            ticks += 1;
            for ev in &events {
                match ev {
                    SynthEvent::Looped => {
                        loops += 1;
                        if loops >= 2 {
                            end_ticks = Some(ticks);
                        }
                    }
                    SynthEvent::Ended => end_ticks = Some(ticks),
                    _ => {}
                }
            }
            if end_ticks.is_some() {
                break;
            }
        }
        Some(end_ticks.unwrap_or(ticks) as f64 / tick_rate)
    }
}

pub trait DevicePlayer: Send {
    fn clock_rate(&self) -> f64;

    fn cycles_per_tick(&self) -> f64;

    fn steps_elapsed(&self) -> u32;

    fn step_rate(&self) -> f64;

    fn steps_per_beat(&self) -> f64;

    fn tick(
        &mut self,
        feedback: &mut TickFeedback,
        config: &PerDeviceSettings,
        events: &mut Vec<SynthEvent>,
    );

    fn tick_rate(&self) -> f64 {
        self.clock_rate() / self.cycles_per_tick()
    }
}

pub fn load_all(bytes: &[u8]) -> Vec<Box<dyn SoundData>> {
    let sdats = nds::Sdat::load_all(bytes);
    if !sdats.is_empty() {
        return sdats
            .into_iter()
            .map(|sdat| -> Box<dyn SoundData> { Box::new(sdat) })
            .collect();
    }
    if let Some(dse) = dse::DseSoundData::load_all(bytes) {
        return vec![Box::new(dse)];
    }
    match gba::GbaRom::parse(bytes) {
        Some(rom) => vec![Box::new(rom)],
        None => Vec::new(),
    }
}
