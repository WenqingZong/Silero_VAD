pub mod utils;

// System libraries.

// Third Party libraries.
use anyhow::bail;
use log::{debug, info};
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
    pub vad_config: VadConfig,

    vad_state: VadState,
    audio_buffer: Vec<f32>,
    redemption_count: usize,
    total_processed_frames: usize,
    current_speech_segments: usize,
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

/// A piece of active voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VadResult {
    pub start_ms: usize,
    pub end_ms: usize,
}

impl VAD {
    /// Construct an VAD, currently only support 8000 and 16000 Hz audio data.
    pub fn new(sample_rate: usize, vad_config: VadConfig) -> anyhow::Result<Self> {
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
            vad_config,
            vad_state: VadState::Silence,
            audio_buffer: Vec::new(),
            redemption_count: 0,
            total_processed_frames: 0,
            current_speech_segments: 0,
            model,
            h_tensor,
            c_tensor,
            sample_rate_tensor,
        })
    }

    /// Get VAD algorithm frame size.
    pub fn frame_size(&self) -> usize {
        self.vad_config.frame_samples
    }

    /// Process one frame, return the probability of this frame is active speech.
    pub fn process_frame(&mut self, data: &[f32]) -> anyhow::Result<f32> {
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

    pub fn process_audio(&mut self, mono_audio: &[f32]) -> Vec<VadResult> {
        self.audio_buffer.extend_from_slice(mono_audio);
        let mut ret = Vec::new();
        let mut start_ms = 0;
        let mut processed_frames = 0;

        // The main algorithm, described here: https://wiki.vad.ricky0123.com/en/docs/user/algorithm
        let num_frames = self.audio_buffer.len() / self.frame_size();
        for i in 0..num_frames {
            let start_idx = i * self.frame_size();
            let end_idx = (i + 1) * self.frame_size();
            let audio_frame = &self.audio_buffer.clone()[start_idx..end_idx];

            let speech_prob = self.process_frame(audio_frame).unwrap();
            debug!("[{}ms -> {}ms]: {}",
                (((self.total_processed_frames + i) * self.frame_size()) as f32 / 8000.0 * 1000.0) as usize,
                (((self.total_processed_frames + i + 1) * self.frame_size()) as f32 / 8000.0 * 1000.0) as usize,
                speech_prob,
            );

            if self.vad_state == VadState::Silence && speech_prob > self.vad_config.positive_speech_threshold {
                start_ms = (((self.total_processed_frames + i) * self.frame_size()) as f32 / 8000.0 * 1000.0) as usize;
                info!("Silence -> Speech detected at {}ms", start_ms);
                self.vad_state = VadState::Speech;
                self.current_speech_segments += 1;
            } else if self.vad_state == VadState::Speech {
                self.current_speech_segments += 1;
                if speech_prob < self.vad_config.negative_speech_threshold {
                    self.redemption_count += 1;
                    if self.redemption_count > self.vad_config.redemption_frames {
                        if self.current_speech_segments >= self.vad_config.min_speech_frames {
                            let end_ms = (((self.total_processed_frames + i + 1) * self.frame_size()) as f32 / 8000.0 * 1000.0) as usize;
                            info!("Speech -> Silence detected at {}ms", end_ms);
                            ret.push(VadResult {
                                start_ms,
                                end_ms,
                            });
                        }
                        self.vad_state = VadState::Silence;
                        self.redemption_count = 0;
                        self.current_speech_segments = 0;
                    }
                } else {
                    self.redemption_count = 0;
                }
            }

            self.total_processed_frames += 1;
            processed_frames += 1;
        }

        // Clear audio buffer.
        if processed_frames > 0 {
            let start_idx = (processed_frames - 1) * self.frame_size();
            let end_idx = processed_frames * self.frame_size();
            drop(self.audio_buffer.drain(start_idx..end_idx));
        }
        ret
    }

    pub fn reset(&mut self) {
        self.h_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.c_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.redemption_count = 0;
        self.vad_state = VadState::Silence;
        self.current_speech_segments = 0;
        self.audio_buffer.clear();
        self.total_processed_frames = 0;
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
