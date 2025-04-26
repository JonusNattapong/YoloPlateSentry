use std::path::Path;
use image::DynamicImage;
use leptess::{tesseract::TessApi, LepTess};
use regex::Regex;
use thiserror::Error;
use tracing::{debug, info};

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("Failed to initialize Tesseract: {0}")]
    TesseractInitError(String),
    #[error("Failed to process image: {0}")]
    ImageProcessError(String),
    #[error("OCR processing error: {0}")]
    ProcessingError(String),
    #[error("Invalid license plate format: {0}")]
    ValidationError(String),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LicensePlateText {
    pub text: String,
    pub confidence: f32,
    pub processed_text: String,  // Cleaned and formatted text
}

pub struct PlateOcr {
    tesseract: LepTess,
    plate_pattern: Regex,
}

impl PlateOcr {
    pub fn new() -> Result<Self, OcrError> {
        info!("Initializing OCR engine");

        // Initialize Tesseract with English language
        let mut tesseract = LepTess::new(None, "eng").map_err(|e| {
            OcrError::TesseractInitError(format!("Failed to initialize Tesseract: {}", e))
        })?;

        // Configure Tesseract for license plate recognition
        tesseract
            .set_variable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
            .map_err(|e| OcrError::TesseractInitError(e.to_string()))?;

        // Compile regex pattern for license plate validation
        // This is a basic pattern - adjust based on your specific license plate format
        let plate_pattern = Regex::new(r"^[A-Z0-9-]{4,10}$").map_err(|e| {
            OcrError::TesseractInitError(format!("Failed to compile regex pattern: {}", e))
        })?;

        Ok(Self {
            tesseract,
            plate_pattern,
        })
    }

    pub fn process_plate(&self, image: &DynamicImage) -> Result<LicensePlateText, OcrError> {
        // Preprocess image for better OCR accuracy
        let processed_image = self.preprocess_image(image)?;

        // Convert image for Tesseract
        let width = processed_image.width() as i32;
        let height = processed_image.height() as i32;
        let bytes = processed_image.to_luma8().into_raw();

        // Set image data
        self.tesseract
            .set_image_from_mem(&bytes, width, height, 1, width)
            .map_err(|e| OcrError::ProcessingError(e.to_string()))?;

        // Perform OCR
        let text = self.tesseract
            .get_utf8_text()
            .map_err(|e| OcrError::ProcessingError(e.to_string()))?;

        let confidence = self.tesseract
            .mean_text_conf() as f32 / 100.0;

        // Post-process and validate the text
        let processed_text = self.postprocess_text(&text)?;

        debug!(
            "OCR Result - Raw: {}, Processed: {}, Confidence: {:.2}",
            text.trim(),
            processed_text,
            confidence
        );

        Ok(LicensePlateText {
            text: text.trim().to_string(),
            confidence,
            processed_text,
        })
    }

    fn preprocess_image(&self, image: &DynamicImage) -> Result<DynamicImage, OcrError> {
        let mut processed = image.clone();

        // Convert to grayscale
        processed = DynamicImage::ImageLuma8(processed.to_luma8());

        // Resize if too small or too large
        if processed.width() < 100 || processed.height() < 30 {
            processed = processed.resize(
                processed.width() * 2,
                processed.height() * 2,
                image::imageops::FilterType::Lanczos3,
            );
        } else if processed.width() > 1000 || processed.height() > 300 {
            processed = processed.resize(
                1000,
                300,
                image::imageops::FilterType::Lanczos3,
            );
        }

        // Enhance contrast
        processed = DynamicImage::ImageLuma8(
            imageproc::contrast::stretch_contrast(&processed.to_luma8(), 50)
        );

        // Apply adaptive thresholding
        let threshold = imageproc::contrast::threshold_adaptive(
            &processed.to_luma8(),
            15,  // block radius
            |x| x as f32,
        );
        processed = DynamicImage::ImageLuma8(threshold);

        Ok(processed)
    }

    fn postprocess_text(&self, text: &str) -> Result<String, OcrError> {
        // Clean up the text
        let processed = text
            .trim()
            .replace(['\n', ' '], "")
            .to_uppercase();

        // Validate against the pattern
        if !self.plate_pattern.is_match(&processed) {
            return Err(OcrError::ValidationError(format!(
                "Text '{}' does not match license plate pattern",
                processed
            )));
        }

        Ok(processed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_initialization() {
        let ocr = PlateOcr::new();
        assert!(ocr.is_ok());
    }

    #[test]
    fn test_text_postprocessing() {
        let ocr = PlateOcr::new().unwrap();
        
        // Test valid plate number
        assert!(ocr.postprocess_text("ABC123").is_ok());
        
        // Test invalid plate number
        assert!(ocr.postprocess_text("!@#$%^").is_err());
    }

    #[test]
    fn test_image_preprocessing() {
        // TODO: Add tests with sample images
    }
}