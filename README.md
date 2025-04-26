# 🚗 YoloPlateSentry

> Smart Vehicle Access Control — Fast, Private, and Affordable

YoloPlateSentry is an intelligent edge AI system for real-time license plate detection and access control. Built with YOLOv8 for detection and Rust for high-performance processing.

## ✨ Features

- **Real-Time License Plate Detection**
  - High-performance detection using YOLOv8
  - CUDA-accelerated inference via ONNX Runtime
  - Support for various camera angles and lighting conditions

- **Accurate OCR Processing**
  - Tesseract-based OCR with custom preprocessing
  - Adaptive image enhancement for better accuracy
  - Configurable license plate format validation

- **Smart Access Control**
  - JSON-based whitelist/blacklist system
  - Multiple access levels (Guest, Staff, VIP)
  - Real-time validation and decision making

- **Instant Notifications**
  - LINE Notify integration
  - Telegram bot support
  - Rich notifications with images and details

## 🛠 Requirements

- CUDA-capable GPU
- Rust toolchain (2021 edition or later)
- Tesseract 4.0+ and development libraries
- CUDA Toolkit 11.0+
- OpenCV (for camera capture)
- Camera device (USB/IP/RTSP)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/YoloPlateSentry.git
cd YoloPlateSentry
```

2. Install system dependencies (Ubuntu example):
```bash
sudo apt update
sudo apt install -y tesseract-ocr libtesseract-dev libleptonica-dev pkg-config
```

3. Download YOLOv8 ONNX model:
```bash
mkdir -p models
# Download YOLOv8n.onnx from YOLO releases or export your own
```

4. Build the project:
```bash
cargo build --release
```

## ⚙️ Configuration

1. Copy the example configuration:
```bash
cp config.json.example config.json
```

2. Edit `config.json` with your settings:
```json
{
    "model_path": "models/yolov8n.onnx",
    "camera_url": "rtsp://camera_ip:554/stream",
    "line_token": "your_line_notify_token",
    "telegram_token": "your_telegram_bot_token",
    "telegram_chat_id": "your_chat_id",
    "whitelist_path": "data/whitelist.json"
}
```

3. Add allowed license plates to `data/whitelist.json`:
```json
[
    "ABC123",
    "XYZ789"
]
```

## 🚀 Usage

1. Start the application:
```bash
cargo run --release
```

2. Monitor the output:
- Check terminal for detection logs
- Watch for notifications in LINE/Telegram
- Images are saved in `detections/` directory

## 🏗 Architecture

```plaintext
📷 Camera Feed
   ↓
🔍 YOLOv8 Detection (CUDA-accelerated)
   ↓
📝 Tesseract OCR
   ↓
✅ Access Validation
   ↓
📱 Notifications
```

## 🔧 Development

### Project Structure
```
.
├── src/                    # Main application code
├── crates/
│   ├── yolo-detector/     # YOLO detection module
│   ├── plate-ocr/         # OCR processing module
│   └── notification/      # Notification services
├── models/                # YOLO model files
├── data/                  # Configuration files
└── detections/           # Saved detection images
```

### Building Modules
```bash
# Build specific module
cargo build -p yolo-detector

# Run tests
cargo test -p plate-ocr

# Check documentation
cargo doc --open
```

## 📋 TODO

- [ ] Add support for multiple camera streams
- [ ] Implement web dashboard
- [ ] Add more notification providers
- [ ] Support custom YOLO models
- [ ] Improve OCR accuracy for different plate styles

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Tesseract OCR by Google
- ONNX Runtime by Microsoft
- Rust community and crate authors
