// System libraries.
use std::path::{Path, PathBuf};

// Third party libraries.
use anyhow::{bail, Result};
use env_logger::Env;
use ndarray::{Array1, Array2, Array3, ArrayBase, Ix1, Ix3, OwnedRepr};
use ndarray_stats::QuantileExt;
use ort::{GraphOptimizationLevel, Session};
use log::{info, error, warn, debug};

// Project libraries.
use silero_vad::utils::{read_audio_file, save_pcm_to_wav};
use silero_vad::{MultiChannelStrategy, VAD, VadConfig};

// fn main() -> Result<()> {
//     // System init.
//     tracing_subscriber::fmt::init();
//     let mut fvad = Fvad::new()
//         .unwrap()
//         .set_mode(Mode::VeryAggressive)
//         .set_sample_rate(SampleRate::Rate16kHz);
//     let sample_rate = 16000;
//     let chunk_size = 30 * sample_rate / 1000;
//     let vad_config = VadConfig::new(sample_rate);
//     dbg!(&vad_config);
//     let mut vad_state = VadState::Silence;
//     let mut redemption_count = 0;
//     let mut current_speech_segments = Vec::new();
//
//     // Load ONNX model.
//     let model_path = PathBuf::from(format!(
//         "{}/models/silero_vad.onnx",
//         env!("CARGO_MANIFEST_DIR")
//     ));
//     let mut vad = VadSession::new(model_path, sample_rate).expect("Cannot create VAD session.");
//
//     // Load audio file.
//     let test_file_path = PathBuf::from(format!(
//         "{}/audio_files/short_test_16k.wav",
//         env!("CARGO_MANIFEST_DIR")
//     ));
//     let (audio, _) =
//         load_wav(&test_file_path).unwrap_or_else(|_| panic!("Cannot find {:?}", &test_file_path));
//     println!("Using {:?} as input.", test_file_path);
//
//     // Inference loop.
//     let num_chunks = audio[0].len() / chunk_size;
//     let mut probs = Vec::new();
//
//     for i in 0..num_chunks {
//         let start_idx = i * chunk_size;
//         let end_idx = (i + 1) * chunk_size;
//         let audio_chunk = &audio[0][start_idx..end_idx];
//
//         // FVAD result, for debug only.
//         let audio_chunk_i16 = convert_vec_f32_to_vec_i16(audio_chunk);
//         let fvad_result = fvad.is_voice_frame(&audio_chunk_i16).unwrap();
//
//         // Actual inference.
//         let speech_prob = vad
//             .process(audio_chunk)
//             .expect("cannot perform VAD forward pass");
//         // println!(
//         //     "[{}ms -> {}ms]: {}, fvad = {}",
//         //     i * 30,
//         //     (i + 1) * 30,
//         //     speech_prob,
//         //     fvad_result
//         // );
//         probs.push(speech_prob);
//
//         // The actual algorithm.
//         if vad_state == VadState::Silence && speech_prob > vad_config.positive_speech_threshold {
//             println!("Silence -> Speech detected at {}ms", i * 30);
//             vad_state = VadState::Speech;
//             current_speech_segments.append(&mut audio_chunk.to_vec());
//         } else if vad_state == VadState::Speech {
//             // Append current segment into buffer.
//             current_speech_segments.append(&mut audio_chunk.to_vec());
//
//             if speech_prob < vad_config.negative_speech_threshold {
//                 redemption_count += 1;
//                 if redemption_count > vad_config.redemption_frames {
//                     if current_speech_segments.len() >= vad_config.min_speech_frames {
//                         handle_speech_segment(
//                             &current_speech_segments,
//                             i,
//                             i - current_speech_segments.len() / vad_config.frame_samples,
//                         );
//                     }
//                     vad_state = VadState::Silence;
//                     redemption_count = 0;
//                     current_speech_segments.clear()
//                 }
//             } else {
//                 redemption_count = 0;
//             }
//         }
//     }
//
//     let stat = Array1::from(probs);
//     dbg!(&stat.min().unwrap());
//     dbg!(&stat.max().unwrap());
//     dbg!(&stat.mean().unwrap());
//     dbg!(&stat.std(0.0));
//
//     Ok(())
// }

fn handle_speech_segment(data: &[f32], end_segment_idx: usize, start_segment_idx: usize) {
    println!(
        "    Detected speech from {}ms -> {}ms",
        start_segment_idx * 30,
        (end_segment_idx + 1) * 30
    );
}

fn main() {
    // Setup logger.
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    // Strategy to deal with stereo audio.
    let multi_channel_strategy = MultiChannelStrategy::Average;

    let desired_sample_rate = 8000;
    let audio = read_audio_file("../audio_files/vad_test_16k.wav", desired_sample_rate, multi_channel_strategy);
    dbg!(audio.len());
    let stat = Array1::from(audio.clone());
    dbg!(&stat.min().unwrap());
    dbg!(&stat.max().unwrap());
    dbg!(&stat.mean().unwrap());
    dbg!(&stat.std(0.0));
    save_pcm_to_wav("./output.wav", audio.clone(), desired_sample_rate as u32, 1).unwrap();
    let vad_config = VadConfig::default();
    let mut vad = VAD::new(desired_sample_rate, vad_config).unwrap();
    let results = vad.process_audio(audio.as_slice());
    dbg!(&results);
}