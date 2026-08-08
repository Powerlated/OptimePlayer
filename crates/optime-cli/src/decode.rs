//! Decodes a recorded audio file to mono f32 at its own sample rate, for the tools that compare
//! rendered songs against real recordings.

use std::path::Path;

pub fn decode_mono(path: &Path) -> Result<(Vec<f32>, f64), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| e.to_string())?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "no default track".to_string())?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| e.to_string())?;

    let mut mono = Vec::new();
    let mut rate = 0f64;
    let mut interleaved: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(Error::ResetRequired) => break,
            Err(e) => return Err(e.to_string()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.to_string()),
        };
        let spec = *decoded.spec();
        rate = f64::from(spec.rate);
        let channels = spec.channels.count().max(1);
        let buf =
            interleaved.get_or_insert_with(|| SampleBuffer::new(decoded.capacity() as u64, spec));
        buf.copy_interleaved_ref(decoded);
        for frame in buf.samples().chunks_exact(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }
    if mono.is_empty() || rate <= 0.0 {
        return Err("decoded no audio".to_string());
    }
    Ok((mono, rate))
}
