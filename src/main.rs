mod utils;

// System libraries.
use std::path::{Path, PathBuf};

// Third party libraries.
use anyhow::{bail, Result};
use env_logger::Env;
use fvad::{Fvad, Mode, SampleRate};
use ndarray::{Array1, Array2, Array3, ArrayBase, Ix1, Ix3, OwnedRepr};
use ndarray_stats::QuantileExt;
use ort::{GraphOptimizationLevel, Session};
use log::{info, error, warn, debug};

// Project libraries.
use crate::utils::{read_audio_file, save_pcm_to_wav};

#[derive(Clone, Copy, Debug)]
struct VadConfig {
    positive_speech_threshold: f32,
    negative_speech_threshold: f32,
    pre_speech_pad_frames: usize,
    redemption_frames: usize,
    frame_samples: usize,
    min_speech_frames: usize,
}

#[derive(Debug)]
struct VadSession {
    model: Session,
    h_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    c_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    sample_rate_tensor: ArrayBase<OwnedRepr<i64>, Ix1>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VadState {
    Speech,
    Silence,
}

impl VadSession {
    pub fn new(model_file_path: impl AsRef<Path>, sample_rate: usize) -> Result<Self> {
        if ![8000_usize, 16000].contains(&sample_rate) {
            bail!("Unsupported sample rate, use 8000 or 16000!");
        }
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_file_path)?;
        let h_tensor = Array3::<f32>::zeros((2, 1, 64));
        let c_tensor = Array3::<f32>::zeros((2, 1, 64));
        let sample_rate_tensor = Array1::from_vec(vec![sample_rate as i64]);
        Ok(Self {
            model,
            h_tensor,
            c_tensor,
            sample_rate_tensor,
        })
    }

    /// Process one frame, return the probability of this frame is active speech.
    pub fn process(&mut self, data: &[f32]) -> Result<f32> {
        let audio_tensor = Array2::from_shape_vec((1, data.len()), data.to_vec())?;
        let result = self.model.run(ort::inputs![
            audio_tensor.view(),
            self.sample_rate_tensor.view(),
            self.h_tensor.view(),
            self.c_tensor.view()
        ]?)?;

        // Update internal state tensors.
        self.h_tensor = result
            .get("hn")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap()
            .to_owned()
            .into_shape((2, 1, 64))
            .expect("Shape mismatch for h_tensor");
        self.c_tensor = result
            .get("cn")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap()
            .to_owned()
            .into_shape((2, 1, 64))
            .expect("Shape mismatch for c_tensor");

        Ok(*result
            .get("output")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap()
            .first()
            .unwrap())
    }

    pub fn reset(&mut self) {
        self.h_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.c_tensor = Array3::<f32>::zeros((2, 1, 64));
    }
}

impl VadConfig {
    pub fn new(sample_rate: usize) -> Self {
        Self {
            // Original implementation: https://github.com/ricky0123/vad/blob/ea584aaf66d9162fb19d9bfba607e264452980c3/packages/_common/src/frame-processor.ts#L52
            positive_speech_threshold: 0.5, // This was 0.5 in the original JS implementation, I've reduced it to 0.3, but still miss some speech segments.
            negative_speech_threshold: 0.35, // This was 0.35 in the original JS implementation.
            pre_speech_pad_frames: 1,
            redemption_frames: 8,
            frame_samples: (30_f32 / 1000_f32 * sample_rate as f32) as usize, // 30ms * sample_rate Hz.
            min_speech_frames: 3,
        }
    }
}

fn load_wav(file_path: impl AsRef<Path>) -> Result<(Vec<Vec<f32>>, i64)> {
    let mut reader = hound::WavReader::open(file_path).unwrap();
    let sample_rate = reader.spec().sample_rate;
    let num_channels = reader.spec().channels;
    let interleave: Vec<f32> = reader
        .samples::<i16>()
        .filter_map(Result::ok)
        .map(|s| s as f32 / i16::MAX as f32)
        .collect();
    let mut audio = Vec::new();
    if num_channels == 1 {
        audio.push(interleave);
    } else if num_channels == 2 {
        audio.push(vec![]);
        audio.push(vec![]);
        for (i, sample) in interleave.iter().enumerate() {
            if i % 2 == 0 {
                audio[0].push(*sample);
            } else {
                audio[1].push(*sample);
            }
        }
    }
    Ok((audio, sample_rate as i64))
}

pub fn convert_vec_f32_to_vec_i16(data: &[f32]) -> Vec<i16> {
    data.iter()
        .map(|&sample| {
            // Clamp the values to the -1.0 to 1.0 range to prevent unexpected behavior
            let clamped_sample = sample.clamp(-1.0, 1.0);
            // Scale the clamped sample to the i16 range
            let scaled_sample = clamped_sample * 32767.0;
            // Convert to i16
            scaled_sample as i16
        })
        .collect()
}

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
    // env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    // let samples = read_audio_file("./audio_files/prodrecall_16k.wav", 44100);
    // dbg!(samples.len());
    // let stat = Array1::from(samples.clone());
    // dbg!(&stat.min().unwrap());
    // dbg!(&stat.max().unwrap());
    // dbg!(&stat.mean().unwrap());
    // dbg!(&stat.std(0.0));
    // save_pcm_to_wav("./output.wav", samples, 44100, 1).unwrap();
    println!("Library is still under development, thank you for your patience");
}