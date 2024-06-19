// System libraries.

// Third party libraries.
use env_logger::Env;
use log::info;

// Project libraries.
use silero_vad::utils::{read_audio_file, save_pcm_to_wav};
use silero_vad::{MultiChannelStrategy, VadSession, VadConfig};

fn main() {
    // Setup logger.
    env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();

    // Strategy to deal with stereo audio.
    let multi_channel_strategy = MultiChannelStrategy::FirstOnly;

    let desired_sample_rate = 16000;
    let audio = read_audio_file("../audio_files/short_test.wav", desired_sample_rate, multi_channel_strategy);

    save_pcm_to_wav("./output.wav", audio.clone(), desired_sample_rate as u32, 1).unwrap();
    let vad_config = VadConfig::new(desired_sample_rate);
    let mut vad = VadSession::new(vad_config).unwrap();

    let chunk_size = 30 * desired_sample_rate / 1000;
    let num_chunks = audio.len() / chunk_size;

    for i in 0..num_chunks {
        let start_idx = i * chunk_size;
        let end_idx = (i + 1) * chunk_size;

        let audio_chunk = &audio[start_idx..end_idx];
        let transition = vad.process(audio_chunk).unwrap();
        if !transition.is_none() {
            info!("[{}ms -> {}ms]: {:?}", i * 30, (i + 1) * 30, transition.unwrap());
        }
    }
}
