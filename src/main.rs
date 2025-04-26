use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;

use yolo_detector::LicensePlateDetector;
use plate_ocr::PlateOcr;
use notification::{NotificationService, DetectionEvent, AccessStatus};

// Configuration structure
#[derive(Debug, serde::Deserialize)]
struct Config {
    model_path: PathBuf,
    camera_url: String,
    line_token: Option<String>,
    telegram_token: Option<String>,
    telegram_chat_id: Option<String>,
    whitelist_path: PathBuf,
}

struct App {
    detector: Arc<LicensePlateDetector>,
    ocr: Arc<PlateOcr>,
    notifier: Arc<NotificationService>,
    whitelist: Arc<Mutex<std::collections::HashSet<String>>>,
}

impl App {
    async fn new(config: Config) -> Result<Self, Box<dyn Error>> {
        // Initialize YOLO detector
        let detector = Arc::new(LicensePlateDetector::new(config.model_path).await?);
        
        // Initialize OCR
        let ocr = Arc::new(PlateOcr::new()?);
        
        // Initialize notification service
        let notifier = Arc::new(NotificationService::new(
            config.line_token,
            config.telegram_token,
            config.telegram_chat_id,
        ));

        // Load whitelist
        let whitelist = Arc::new(Mutex::new(load_whitelist(&config.whitelist_path)?));

        Ok(Self {
            detector,
            ocr,
            notifier,
            whitelist,
        })
    }

    async fn process_frame(&self, frame: image::DynamicImage) -> Result<(), Box<dyn Error>> {
        // Detect license plates in the frame
        let detections = self.detector.detect_license_plate(&frame).await?;

        for bbox in detections {
            // Extract the license plate region
            let plate_image = frame.crop(
                bbox.x_min as u32,
                bbox.y_min as u32,
                (bbox.x_max - bbox.x_min) as u32,
                (bbox.y_max - bbox.y_min) as u32,
            );

            // Perform OCR on the plate
            let plate_text = self.ocr.process_plate(&plate_image)?;

            // Check if the plate is in the whitelist
            let access_status = {
                let whitelist = self.whitelist.lock().await;
                if whitelist.contains(&plate_text.processed_text) {
                    AccessStatus::Allowed
                } else {
                    AccessStatus::Suspicious
                }
            };

            // Create detection event
            let event = DetectionEvent {
                timestamp: chrono::Utc::now(),
                plate_number: plate_text.processed_text,
                confidence: plate_text.confidence,
                image_path: save_detection_image(&frame, &bbox)?,
                access_status,
            };

            // Send notification if suspicious
            if matches!(event.access_status, AccessStatus::Suspicious) {
                if let Err(e) = self.notifier.send_alert(&event).await {
                    error!("Failed to send alert: {}", e);
                }
            }

            info!("Processed plate: {:?}", event);
        }

        Ok(())
    }

    async fn run_camera_loop(&self, camera_url: String) -> Result<(), Box<dyn Error>> {
        // TODO: Implement camera capture loop using OpenCV or similar
        // For now, just log that we would process frames
        info!("Would process camera feed from: {}", camera_url);
        
        // Placeholder for camera loop
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }
}

fn load_whitelist(path: &PathBuf) -> Result<std::collections::HashSet<String>, Box<dyn Error>> {
    let content = std::fs::read_to_string(path)?;
    let plates: Vec<String> = serde_json::from_str(&content)?;
    Ok(plates.into_iter().collect())
}

fn save_detection_image(
    frame: &image::DynamicImage,
    bbox: &yolo_detector::BoundingBox,
) -> Result<String, Box<dyn Error>> {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S%.3f");
    let path = format!("detections/{}.jpg", timestamp);
    
    // Ensure detections directory exists
    std::fs::create_dir_all("detections")?;
    
    // Draw bounding box and save image
    let mut img_with_box = frame.clone();
    imageproc::drawing::draw_hollow_rect_mut(
        &mut img_with_box,
        imageproc::rect::Rect::at(bbox.x_min as i32, bbox.y_min as i32)
            .of_size(
                (bbox.x_max - bbox.x_min) as u32,
                (bbox.y_max - bbox.y_min) as u32,
            ),
        image::Rgba([255, 0, 0, 255]),
    );
    img_with_box.save(&path)?;
    
    Ok(path)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting YoloPlateSentry...");

    // Load configuration
    let config: Config = {
        let config_text = std::fs::read_to_string("config.json")?;
        serde_json::from_str(&config_text)?
    };

    // Initialize application
    let app = App::new(config.clone()).await?;

    // Run the main camera loop
    app.run_camera_loop(config.camera_url).await?;

    Ok(())
}