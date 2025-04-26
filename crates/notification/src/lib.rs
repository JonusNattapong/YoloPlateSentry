use std::path::Path;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use tracing::{debug, error, info};

#[derive(Debug, Error)]
pub enum NotificationError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("API request failed: {0}")]
    ApiError(String),
    #[error("Failed to process image: {0}")]
    ImageError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionEvent {
    pub timestamp: DateTime<Utc>,
    pub plate_number: String,
    pub confidence: f32,
    pub image_path: String,
    pub access_status: AccessStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessStatus {
    Allowed,
    Denied,
    Suspicious,
}

pub struct NotificationService {
    line_token: Option<String>,
    telegram_token: Option<String>,
    telegram_chat_id: Option<String>,
}

impl NotificationService {
    pub fn new(
        line_token: Option<String>,
        telegram_token: Option<String>,
        telegram_chat_id: Option<String>,
    ) -> Self {
        Self {
            line_token,
            telegram_token,
            telegram_chat_id,
        }
    }

    pub async fn send_alert(&self, event: &DetectionEvent) -> Result<(), NotificationError> {
        let message = self.format_message(event);
        let image_path = Path::new(&event.image_path);

        info!("Sending alert for plate: {}", event.plate_number);

        // Try sending through LINE Notify
        if let Some(token) = &self.line_token {
            match self.send_line_notify(&message, Some(image_path)).await {
                Ok(_) => debug!("Successfully sent LINE notification"),
                Err(e) => error!("Failed to send LINE notification: {}", e),
            }
        }

        // Try sending through Telegram
        if let (Some(token), Some(chat_id)) = (&self.telegram_token, &self.telegram_chat_id) {
            match self.send_telegram(&message, Some(image_path)).await {
                Ok(_) => debug!("Successfully sent Telegram message"),
                Err(e) => error!("Failed to send Telegram message: {}", e),
            }
        }

        Ok(())
    }

    fn format_message(&self, event: &DetectionEvent) -> String {
        let status = match event.access_status {
            AccessStatus::Allowed => "✅ Allowed",
            AccessStatus::Denied => "❌ Denied",
            AccessStatus::Suspicious => "⚠️ Suspicious",
        };

        format!(
            "🚗 License Plate Detection\n\n\
             Plate: {}\n\
             Status: {}\n\
             Confidence: {:.1}%\n\
             Time: {}",
            event.plate_number,
            status,
            event.confidence * 100.0,
            event.timestamp.format("%Y-%m-%d %H:%M:%S"),
        )
    }

    async fn send_line_notify(
        &self,
        message: &str,
        image_path: Option<&Path>,
    ) -> Result<(), NotificationError> {
        let token = self.line_token.as_ref().ok_or_else(|| {
            NotificationError::ConfigError("LINE Notify token not configured".into())
        })?;

        let client = reqwest::Client::new();
        let mut form = reqwest::multipart::Form::new().text("message", message.to_string());

        // Add image if provided
        if let Some(path) = image_path {
            let image_data = tokio::fs::read(path)
                .await
                .map_err(|e| NotificationError::ImageError(e.to_string()))?;

            form = form.part(
                "imageFile",
                reqwest::multipart::Part::bytes(image_data)
                    .file_name("detection.jpg")
                    .mime_str("image/jpeg")
                    .map_err(|e| NotificationError::ImageError(e.to_string()))?,
            );
        }

        // Send request to LINE Notify API
        let response = client
            .post("https://notify-api.line.me/api/notify")
            .header("Authorization", format!("Bearer {}", token))
            .multipart(form)
            .send()
            .await
            .map_err(|e| NotificationError::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".into());
            return Err(NotificationError::ApiError(format!(
                "LINE Notify API error: {}",
                error_text
            )));
        }

        Ok(())
    }

    async fn send_telegram(
        &self,
        message: &str,
        image_path: Option<&Path>,
    ) -> Result<(), NotificationError> {
        let token = self.telegram_token.as_ref().ok_or_else(|| {
            NotificationError::ConfigError("Telegram token not configured".into())
        })?;

        let chat_id = self.telegram_chat_id.as_ref().ok_or_else(|| {
            NotificationError::ConfigError("Telegram chat ID not configured".into())
        })?;

        let client = reqwest::Client::new();
        
        // Send image with caption if provided
        if let Some(path) = image_path {
            let image_data = tokio::fs::read(path)
                .await
                .map_err(|e| NotificationError::ImageError(e.to_string()))?;

            let form = reqwest::multipart::Form::new()
                .text("chat_id", chat_id.clone())
                .text("caption", message.to_string())
                .part(
                    "photo",
                    reqwest::multipart::Part::bytes(image_data)
                        .file_name("detection.jpg")
                        .mime_str("image/jpeg")
                        .map_err(|e| NotificationError::ImageError(e.to_string()))?,
                );

            let response = client
                .post(format!(
                    "https://api.telegram.org/bot{}/sendPhoto",
                    token
                ))
                .multipart(form)
                .send()
                .await
                .map_err(|e| NotificationError::ApiError(e.to_string()))?;

            if !response.status().is_success() {
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".into());
                return Err(NotificationError::ApiError(format!(
                    "Telegram API error: {}",
                    error_text
                )));
            }
        } else {
            // Send text message only
            let response = client
                .post(format!(
                    "https://api.telegram.org/bot{}/sendMessage",
                    token
                ))
                .form(&serde_json::json!({
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                }))
                .send()
                .await
                .map_err(|e| NotificationError::ApiError(e.to_string()))?;

            if !response.status().is_success() {
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".into());
                return Err(NotificationError::ApiError(format!(
                    "Telegram API error: {}",
                    error_text
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_formatting() {
        let service = NotificationService::new(None, None, None);
        let event = DetectionEvent {
            timestamp: Utc::now(),
            plate_number: "ABC123".into(),
            confidence: 0.95,
            image_path: "test.jpg".into(),
            access_status: AccessStatus::Suspicious,
        };

        let message = service.format_message(&event);
        assert!(message.contains("ABC123"));
        assert!(message.contains("95.0%"));
        assert!(message.contains("⚠️ Suspicious"));
    }

    #[tokio::test]
    async fn test_line_notification() {
        // TODO: Add integration tests with mock server
    }

    #[tokio::test]
    async fn test_telegram_notification() {
        // TODO: Add integration tests with mock server
    }
}