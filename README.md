# 🎯 Carrom Pool ML Bot

An advanced machine learning bot that plays Carrom Pool using computer vision, reinforcement learning, and ADB automation. Perfect for both carrom experts and complete beginners!

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Android-brightgreen.svg)

## 🌟 Features

### 🤖 AI-Powered Gameplay
- **Computer Vision**: Real-time board state detection using OpenCV
- **Machine Learning**: Random Forest classifier for strategic decision making
- **Physics Simulation**: Accurate shot prediction and collision modeling
- **Expert Learning**: Learn from professional gameplay videos

### 🎮 Multiple Operation Modes
- **Expert Mode**: Learn from professional videos (perfect for beginners!)
- **Train Mode**: Self-learning through reinforcement learning
- **Play Mode**: Autonomous gameplay using trained models
- **Collect Mode**: Data collection from human gameplay

### 📱 Android Integration
- **ADB Automation**: Direct device control via Android Debug Bridge
- **Screen Capture**: Real-time screenshot analysis
- **Touch Simulation**: Precise shot execution
- **Multi-Device Support**: Works with any Android device

## 🚀 Quick Start

### For Complete Beginners (No Carrom Skills Needed!)

```bash
# 1. Clone and setup
git clone <repository-url>
cd "Ai carrom"
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Learn from expert videos (no Android device needed!)
python3 main.py --mode expert --video expert_gameplay.mp4 --debug

# 3. Train the AI with built-in strategies
python3 main.py --mode train --episodes 100 --debug

# 4. Connect Android device and play!
adb devices  # Verify device connection
python3 main.py --mode play --episodes 5 --debug
```

### For Experienced Users

```bash
# Setup
git clone <repository-url>
cd "Ai carrom"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure for your device
nano config.yaml  # Adjust screen resolution and game area

# Start data collection
python3 main.py --mode collect --episodes 50 --debug

# Train model
python3 main.py --mode train --episodes 200 --debug

# Play game
python3 main.py --mode play --episodes 10 --debug
```

## 📋 Prerequisites

### Required
- **Python 3.9+** with pip
- **ADB (Android Debug Bridge)** installed
- **Android device** with USB debugging enabled
- **Carrom Pool app** by Miniclip installed on device

### Optional
- Expert gameplay videos (for learning mode)
- Multiple Android devices (for advanced training)

## 🛠️ Installation

### 1. System Dependencies

**macOS:**
```bash
# Install ADB via Homebrew
brew install android-platform-tools

# Or install Android Studio for full SDK
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install android-tools-adb python3-dev python3-pip
```

**Windows:**
```bash
# Download ADB from Android Developer website
# Or install via scoop: scoop install adb
```

### 2. Python Environment

```bash
# Clone repository
git clone <repository-url>
cd "Ai carrom"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Android Setup

```bash
# Enable Developer Options on Android device
# Enable USB Debugging
# Connect device via USB
# Trust computer when prompted

# Verify connection
adb devices
# Should show: List of devices attached
#              DEVICE_ID    device
```

## 🎯 Usage Modes

### 🌟 Expert Mode (Recommended for Beginners)

Learn from professional gameplay without needing carrom skills:

```bash
# Analyze expert video and train automatically
python3 main.py --mode expert --video expert_gameplay.mp4 --debug

# What it does:
# 1. Extracts game states from video
# 2. Analyzes expert strategies by game phase
# 3. Generates training data
# 4. Trains ML model automatically
```

### 🤖 Train Mode

Self-learning through reinforcement learning:

```bash
# Train with built-in expert strategies
python3 main.py --mode train --episodes 100 --debug

# Advanced training with custom episodes
python3 main.py --mode train --episodes 500 --config custom_config.yaml
```

### 🎮 Play Mode

Autonomous gameplay using trained models:

```bash
# Play with trained model
python3 main.py --mode play --episodes 10 --debug

# Continuous play
python3 main.py --mode play --episodes 999 --device DEVICE_ID
```

### 📊 Collect Mode

Data collection from human gameplay:

```bash
# Collect training data while you play
python3 main.py --mode collect --episodes 50 --debug

# Collect from specific device
python3 main.py --mode collect --device DEVICE_ID --debug
```

## ⚙️ Configuration

### Basic Configuration (`config.yaml`)

```yaml
# Device Settings
device:
  screen_resolution: [1080, 1920]  # Adjust to your device
  game_area: [100, 300, 980, 1620]  # Game board boundaries

# ML Settings
model:
  rl:
    epsilon_start: 1.0      # Exploration rate
    learning_rate: 0.001    # Learning speed
    memory_size: 10000      # Training memory

# Training Settings
training:
  episodes: 1000           # Training episodes
  save_frequency: 100      # Model save interval
```

### Advanced Configuration

See `config.yaml` for complete configuration options including:
- Computer vision parameters
- Physics simulation settings
- Detection thresholds
- Training hyperparameters

## 📁 Project Structure

```
Ai carrom/
├── src/
│   ├── device/              # ADB automation
│   │   └── adb_controller.py
│   ├── vision/              # Computer vision
│   │   └── game_detector.py
│   ├── ml/                  # Machine learning
│   │   ├── strategy_agent.py
│   │   └── board_classifier.py
│   ├── game/                # Game logic
│   │   ├── physics_simulator.py
│   │   └── game_state.py
│   ├── utils/               # Utilities
│   │   ├── config_loader.py
│   │   ├── logger_setup.py
│   │   └── screenshot_manager.py
│   └── carrom_bot.py        # Main bot class
├── config.yaml             # Configuration
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md              # This file
└── README_BEGINNER.md     # Beginner's guide
```

## 🧠 How It Works

### 1. Computer Vision Pipeline
- **Screenshot Capture**: Real-time screen capture via ADB
- **Board Detection**: HSV color filtering and contour detection
- **Piece Recognition**: Template matching and feature extraction
- **State Analysis**: Board state classification and piece counting

### 2. Machine Learning Strategy
- **Feature Extraction**: Convert visual board state to numerical features
- **Strategy Agent**: Random Forest classifier for decision making
- **Action Space**: Continuous striker position, angle, and power
- **Learning**: Experience replay and incremental improvement

### 3. Physics Simulation
- **Collision Detection**: Accurate piece-to-piece and piece-to-wall collisions
- **Trajectory Prediction**: Shot outcome forecasting
- **Friction Modeling**: Realistic piece movement simulation
- **Shot Validation**: Legal move verification

### 4. Game Execution
- **Turn Detection**: Automated turn recognition
- **Shot Calculation**: Optimal striker positioning and power
- **Touch Automation**: Precise ADB touch commands
- **Result Analysis**: Shot success evaluation and learning

## 📈 Performance Monitoring

### Training Progress
```bash
# Monitor training logs
tail -f logs/carrom_bot.log

# Check model improvements
grep "accuracy" logs/carrom_bot.log
grep "win_rate" logs/carrom_bot.log
```

### Game Statistics
```bash
# View bot performance
python3 -c "
from src.carrom_bot import CarromBot
from src.utils.config_loader import ConfigLoader
config = ConfigLoader.load_config('config.yaml')
bot = CarromBot(config)
stats = bot.get_statistics()
print(f'Games played: {stats[\"games_played\"]}')
print(f'Win rate: {stats[\"win_rate\"]:.2%}')
print(f'Average shots: {stats[\"avg_shots_per_game\"]}')
"
```

## 🔧 Troubleshooting

### Common Issues

**"No devices found"**
```bash
adb kill-server
adb start-server
adb devices
```

**Computer vision not working**
- Adjust color ranges in `config.yaml`
- Check lighting conditions
- Verify screen resolution settings

**Poor shot accuracy**
- Collect more training data
- Increase training episodes
- Adjust physics parameters

**Import errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

### Debug Mode

Enable detailed logging:
```bash
python3 main.py --mode play --debug
```

Check debug screenshots:
```bash
ls screenshots/debug/
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Miniclip** for creating Carrom Pool
- **OpenCV** community for computer vision tools
- **scikit-learn** for machine learning capabilities
- **Android** team for ADB tools

## 🔗 Links

- [Carrom Pool on Google Play](https://play.google.com/store/apps/details?id=com.miniclip.carrom)
- [ADB Documentation](https://developer.android.com/studio/command-line/adb)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

---

**Made with ❤️ for the carrom community**

*Perfect for both beginners who want to learn and experts who want to dominate!* 