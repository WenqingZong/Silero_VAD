pub mod utils;

// System libraries.

use std::path::Path;
use anyhow::bail;
// Third Party libraries.
use ndarray::{Array1, Array2, Array3, ArrayBase, Ix1, Ix3, OwnedRepr};
use ort::{GraphOptimizationLevel, Session};

// Project libraries.
pub use utils::*;

/// Define all hyper parameters will be used by VAD inference algorithm.
#[derive(Clone, Copy, Debug)]
pub struct VadConfig {
    pub positive_speech_threshold: f32,
    pub negative_speech_threshold: f32,
    pub pre_speech_pad_frames: usize,
    pub redemption_frames: usize,
    pub frame_samples: usize,
    pub min_speech_frames: usize,
}

/// The voice activity detector.
#[derive(Debug)]
pub struct VAD {
    model: Session,
    h_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    c_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    sample_rate_tensor: ArrayBase<OwnedRepr<i64>, Ix1>,
}

/// Detection result of a piece of audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadState {
    Speech,
    Silence,
}

impl VAD {
    pub fn new(sample_rate: usize) -> anyhow::Result<Self> {
        if ![8000_usize, 16000].contains(&sample_rate) {
            bail!("Unsupported sample rate, use 8000 or 16000!");
        }
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/models/silero_vad.onnx"))?;
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
    pub fn process(&mut self, data: &[f32]) -> anyhow::Result<f32> {
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

impl Default for VadConfig {
    fn default() -> Self {
        // Value copied from https://github.com/ricky0123/vad/blob/ea584aaf66d9162fb19d9bfba607e264452980c3/packages/_common/src/frame-processor.ts#L52
        Self {
            positive_speech_threshold: 0.5,
            negative_speech_threshold: 0.35,
            pre_speech_pad_frames: 1,
            redemption_frames: 8,
            frame_samples: 1536,
            min_speech_frames: 3,
        }
    }
}

impl VadConfig {
    pub fn new() -> Self {
        Self::default()
    }
}
