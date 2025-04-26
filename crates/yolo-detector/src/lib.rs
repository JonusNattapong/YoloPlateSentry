use std::path::Path;
use image::DynamicImage;
use ndarray::{Array, ArrayView, Axis, Dim};
use ort::{
    Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder,
    Value, ValueRef,
};
use thiserror::Error;
use tracing::{debug, info};

#[derive(Debug, Error)]
pub enum DetectorError {
    #[error("Failed to load YOLO model: {0}")]
    ModelLoadError(String),
    #[error("Failed to process image: {0}")]
    ImageProcessError(String),
    #[error("Inference error: {0}")]
    InferenceError(String),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoundingBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub confidence: f32,
}

pub struct LicensePlateDetector {
    session: Session,
    input_name: String,
    output_name: String,
}

const INPUT_HEIGHT: u32 = 640;
const INPUT_WIDTH: u32 = 640;
const CONFIDENCE_THRESHOLD: f32 = 0.5;
const IOU_THRESHOLD: f32 = 0.5;

impl LicensePlateDetector {
    pub async fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, DetectorError> {
        info!("Initializing YOLO detector with model: {:?}", model_path.as_ref());

        // Initialize ONNX Runtime environment with CUDA provider
        let environment = Environment::builder()
            .with_name("YoloPlateSentry")
            .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
            .build()
            .map_err(|e| DetectorError::ModelLoadError(e.to_string()))?;

        // Create session
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .with_model_from_file(model_path)
            .map_err(|e| DetectorError::ModelLoadError(e.to_string()))?;

        // Get input and output names
        let input_name = session
            .inputs[0]
            .name
            .clone()
            .ok_or_else(|| DetectorError::ModelLoadError("Failed to get input name".into()))?;

        let output_name = session
            .outputs[0]
            .name
            .clone()
            .ok_or_else(|| DetectorError::ModelLoadError("Failed to get output name".into()))?;

        Ok(Self {
            session,
            input_name,
            output_name,
        })
    }

    pub async fn detect_license_plate(&self, image: &DynamicImage) -> Result<Vec<BoundingBox>, DetectorError> {
        // Preprocess image
        let input_tensor = self.preprocess_image(image)?;
        
        // Run inference
        let outputs = self.session
            .run([input_tensor])
            .map_err(|e| DetectorError::InferenceError(e.to_string()))?;

        // Post-process output
        let boxes = self.postprocess_output(&outputs[0])?;
        
        debug!("Detected {} license plates", boxes.len());
        Ok(boxes)
    }

    fn preprocess_image(&self, image: &DynamicImage) -> Result<Value, DetectorError> {
        // Resize image
        let resized = image::DynamicImage::ImageRgba8(
            image.resize_exact(INPUT_WIDTH, INPUT_HEIGHT, image::imageops::FilterType::Triangle)
                .to_rgba8()
        );

        // Convert to float32 array and normalize
        let mut input_tensor = vec![0.0f32; (INPUT_HEIGHT * INPUT_WIDTH * 3) as usize];
        
        for (i, pixel) in resized.to_rgb8().pixels().enumerate() {
            // Normalize to [0, 1] and convert to RGB
            input_tensor[i * 3] = pixel[0] as f32 / 255.0;
            input_tensor[i * 3 + 1] = pixel[1] as f32 / 255.0;
            input_tensor[i * 3 + 2] = pixel[2] as f32 / 255.0;
        }

        // Create ONNX tensor
        let array = Array::from_shape_vec(
            (1, 3, INPUT_HEIGHT as usize, INPUT_WIDTH as usize),
            input_tensor
        ).map_err(|e| DetectorError::ImageProcessError(e.to_string()))?;

        Value::from_array(array)
            .map_err(|e| DetectorError::ImageProcessError(e.to_string()))
    }

    fn postprocess_output(&self, output: &ValueRef) -> Result<Vec<BoundingBox>, DetectorError> {
        let array = output
            .try_extract()
            .map_err(|e| DetectorError::InferenceError(e.to_string()))?;

        let shape = array.shape();
        if shape.len() != 3 {
            return Err(DetectorError::InferenceError(
                "Unexpected output shape".into(),
            ));
        }

        let mut boxes = Vec::new();
        let predictions = array.slice(s![0, .., ..]);

        // Extract boxes and scores
        for i in 0..predictions.shape()[0] {
            let confidence = predictions[[i, 4]];
            if confidence > CONFIDENCE_THRESHOLD {
                let x_center = predictions[[i, 0]];
                let y_center = predictions[[i, 1]];
                let width = predictions[[i, 2]];
                let height = predictions[[i, 3]];

                // Convert to corner coordinates
                let x_min = x_center - width / 2.0;
                let y_min = y_center - height / 2.0;
                let x_max = x_center + width / 2.0;
                let y_max = y_center + height / 2.0;

                boxes.push(BoundingBox {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    confidence,
                });
            }
        }

        // Apply NMS
        boxes = self.non_max_suppression(boxes, IOU_THRESHOLD);

        Ok(boxes)
    }

    fn non_max_suppression(&self, mut boxes: Vec<BoundingBox>, iou_threshold: f32) -> Vec<BoundingBox> {
        boxes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        let mut keep = vec![true; boxes.len()];

        for i in 0..boxes.len() {
            if !keep[i] {
                continue;
            }

            for j in (i + 1)..boxes.len() {
                if !keep[j] {
                    continue;
                }

                if self.calculate_iou(&boxes[i], &boxes[j]) > iou_threshold {
                    keep[j] = false;
                }
            }
        }

        boxes
            .into_iter()
            .zip(keep)
            .filter_map(|(bbox, keep)| if keep { Some(bbox) } else { None })
            .collect()
    }

    fn calculate_iou(&self, box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        let x_left = box1.x_min.max(box2.x_min);
        let y_top = box1.y_min.max(box2.y_min);
        let x_right = box1.x_max.min(box2.x_max);
        let y_bottom = box1.y_max.min(box2.y_max);

        if x_right < x_left || y_bottom < y_top {
            return 0.0;
        }

        let intersection_area = (x_right - x_left) * (y_bottom - y_top);
        let box1_area = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min);
        let box2_area = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min);
        
        intersection_area / (box1_area + box2_area - intersection_area)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_detector_initialization() {
        // TODO: Add tests with a small test model
    }

    #[tokio::test]
    async fn test_license_plate_detection() {
        // TODO: Add tests with sample images
    }
}