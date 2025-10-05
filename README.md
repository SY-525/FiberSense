# FiberSense

A machine learning application for fiber analysis using an ensemble deep learning model.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Checkpoint](#model-checkpoint)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

FiberSense is a web-based application that uses deep learning to analyze fiber samples. The project combines a React frontend with a Python backend powered by PyTorch for machine learning inference.

## Features

- Deep learning-based fiber classification
- Ensemble model for improved accuracy
- User-friendly web interface
- Real-time prediction capabilities

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 14+** and npm - [Download Node.js](https://nodejs.org/)
- **Git** - [Download Git](https://git-scm.com/downloads)
- **Git LFS** - Required for downloading the model checkpoint
  ```bash
  # Install Git LFS
  # macOS
  brew install git-lfs
  
  # Windows (using Git for Windows)
  # Git LFS is included
  
  # Linux
  sudo apt-get install git-lfs
  
  # Initialize Git LFS
  git lfs install
  ```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SY-525/FiberSense.git
cd FiberSense
```

### 2. Set Up Python Environment

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

**Note:** If you don't have a `requirements.txt` file yet, create one with the following common dependencies:

```txt
torch>=2.0.0
torchvision>=0.15.0
flask>=2.3.0
numpy>=1.24.0
pillow>=9.5.0
```

### 3. Set Up Frontend

Install Node.js dependencies:

```bash
npm install
```

## Model Checkpoint

The trained model checkpoint (`ensemble_checkpoint_epoch_70.pth`) is stored using Git LFS due to its large size (512MB).

### Downloading the Checkpoint

When you clone the repository with Git LFS installed, the checkpoint should download automatically. If it doesn't, run:

```bash
git lfs pull
```

### Verifying the Checkpoint

Check that the checkpoint file is present and has the correct size:

```bash
# macOS/Linux
ls -lh ensemble_checkpoint_epoch_70.pth

# Windows
dir ensemble_checkpoint_epoch_70.pth
```

The file should be approximately 512MB. If it's only a few KB, it means Git LFS didn't download it properly.

## Usage

### Running the Backend (Python/Flask)

1. Make sure your virtual environment is activated
2. Start the Flask server:

```bash
python app.py
```

The backend should start running on `http://localhost:5000` (or another port specified in `app.py`).

### Running the Frontend (React)

In a new terminal window:

```bash
npm start
```

The frontend will open in your browser at `http://localhost:3000`.

### Making Predictions

1. Open the application in your browser
2. Upload a fiber image
3. Click "Analyze" or "Predict"
4. View the classification results

## Project Structure

```
FiberSense/
├── app.py                              # Flask backend server
├── model.py                            # Model architecture definition
├── ensemble_checkpoint_epoch_70.pth    # Trained model weights (Git LFS)
├── package.json                        # Node.js dependencies
├── package-lock.json                   # Locked Node.js dependencies
├── README.md                           # This file
├── public/                             # Static files
├── src/                                # React source code
│   ├── components/                     # React components
│   ├── App.js                          # Main React application
│   └── ...
└── venv/                               # Python virtual environment (not in git)
```

## Technologies Used

### Backend
- **Python** - Programming language
- **PyTorch** - Deep learning framework
- **Flask** - Web framework for API

### Frontend
- **React** - JavaScript library for UI
- **Node.js** - JavaScript runtime
- **npm** - Package manager

### Model
- **Ensemble Deep Learning Model** - Combines multiple models for better accuracy
- Trained for 70 epochs
- Checkpoint size: 512MB

## Troubleshooting

### Git LFS Issues

**Problem:** Checkpoint file is only a few KB

**Solution:**
```bash
git lfs install
git lfs pull
```

### Python Import Errors

**Problem:** `ModuleNotFoundError`

**Solution:** Make sure your virtual environment is activated and all dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Port Already in Use

**Problem:** Port 5000 or 3000 is already in use

**Solution:** 
- Kill the process using that port, or
- Change the port in your configuration files

### Model Loading Errors

**Problem:** Can't load the checkpoint file

**Solution:**
- Verify the file exists and is ~512MB
- Check that the model architecture in `model.py` matches the checkpoint
- Ensure you have enough RAM (model requires significant memory)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the LICENSE file for details.

## Acknowledgments

- Developed as part of the iGEM 2025 competition
- Team GEMS-Taiwan

## Contact

For questions or support, please open an issue on GitHub.
