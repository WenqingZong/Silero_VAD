pub mod utils;

// System libraries.

// Third Party libraries.
use anyhow::{Result, bail};
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
    pub sample_rate: usize,
    pub min_speech_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        // Value copied from https://github.com/ricky0123/vad/blob/ea584aaf66d9162fb19d9bfba607e264452980c3/packages/_common/src/frame-processor.ts#L52
        Self {
            positive_speech_threshold: 0.5,
            negative_speech_threshold: 0.35,
            pre_speech_pad_frames: 1,
            redemption_frames: 20,
            sample_rate: 16000,
            min_speech_frames: 3,
        }
    }
}

impl VadConfig {
    pub fn get_frame_samples(&self) -> usize {
        (30_f32 / 1000_f32 * self.sample_rate as f32) as usize // 30ms * sample_rate Hz
    }

    pub fn new(sample_rate: usize) -> Self {
        let mut ret = Self::default();
        ret.sample_rate = sample_rate;
        ret
    }
}

/// The voice activity detector.
#[derive(Debug)]
pub struct VadSession {
    config: VadConfig,
    model: Session,
    h_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    c_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    sample_rate_tensor: ArrayBase<OwnedRepr<i64>, Ix1>,
    state: VadState,
    current_speech: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VadState {
    Speech {
        min_frames_passed: bool,
        redemption_count: usize,
    },
    Silence,
}

/// Detection result of a piece of audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadTransition {
    SpeechStart,
    SpeechEnd,
}

impl VadSession {
    /// Construct an VAD, currently only support 8000 and 16000 Hz audio data.
    pub fn new(config: VadConfig) -> Result<Self> {
        if ![8000_usize, 16000].contains(&config.sample_rate) {
            bail!("Unsupported sample rate, use 8000 or 16000!");
        }
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(format!(
                "{}/models/silero_vad.onnx",
                env!("CARGO_MANIFEST_DIR")
            ))?;
        let h_tensor = Array3::<f32>::zeros((2, 1, 64));
        let c_tensor = Array3::<f32>::zeros((2, 1, 64));
        let sample_rate_tensor = Array1::from_vec(vec![config.sample_rate as i64]);

        Ok(Self {
            config,
            model,
            h_tensor,
            c_tensor,
            sample_rate_tensor,
            state: VadState::Silence,
            current_speech: vec![],
        })
    }

    /// Advance the VAD state machine with an audio frame (should be 30ms).
    /// Return indicates if a transition from speech to silence (or silence to speech) occurred.
    ///
    /// Important: don't implement your own end pointing logic.
    /// Instead, when a `SpeechEnd` is returned, you can use the `get_current_speech()` method to retrieve the audio.
    pub fn process(&mut self, audio_frame: &[f32]) -> Result<Option<VadTransition>> {
        let audio_tensor = Array2::from_shape_vec((1, audio_frame.len()), audio_frame.to_vec())?;
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
            .expect("Shape mismatch for h_tensor");

        let prob = *result
            .get("output")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap()
            .first()
            .unwrap();

        let mut vad_change = None;

        match self.state {
            VadState::Silence => {
                if prob > self.config.positive_speech_threshold {
                    self.state = VadState::Speech {
                        min_frames_passed: false,
                        redemption_count: 0,
                    };
                    self.current_speech.clear();
                    self.current_speech.extend_from_slice(audio_frame);
                } else {
                    self.state = VadState::Silence
                }
            },
            VadState::Speech { ref mut min_frames_passed, ref mut redemption_count, } => {
                self.current_speech.extend_from_slice(audio_frame);
                if !*min_frames_passed && self.current_speech.len() >= self.config.min_speech_frames {
                    *min_frames_passed = true;
                    vad_change = Some(VadTransition::SpeechStart)
                }

                if prob < self.config.negative_speech_threshold {
                    *redemption_count += 1;
                    if *redemption_count > self.config.redemption_frames {
                        if *min_frames_passed {
                            vad_change = Some(VadTransition::SpeechEnd);
                        }
                        self.state = VadState::Silence
                    }
                } else {
                    *redemption_count = 0;
                }
            }
        };

        Ok(vad_change)
    }

    pub fn get_current_speech(&self) -> &[f32] {
        &self.current_speech
    }

    pub fn is_speaking(&self) -> bool {
        match self.state {
            VadState::Speech { min_frames_passed , .. } if min_frames_passed => true,
            _ => false,
        }
    }

    pub fn reset(&mut self) {
        self.h_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.c_tensor = Array3::<f32>::zeros((2, 1, 64));
    }
}
