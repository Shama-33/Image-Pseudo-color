# Context-Based Pseudo Coloring

A comprehensive image processing application that transforms grayscale images into color images using reference-based pseudocoloring techniques. This project employs advanced object matching and histogram matching algorithms to achieve accurate and visually appealing colorization.

![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.x-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)


## Overview

Pseudo-coloring is a powerful technique in digital image processing that transforms grayscale images into color images by mapping intensity levels to colors based on a reference image. This project provides multiple approaches to achieve context-aware colorization:

- **Global Pseudocoloring**: Colors the entire image based on global characteristics
- **Local Pseudocoloring**: Colors specific regions of interest (ROI)
- **Object-Based Coloring**: Matches and colors individual objects
- **Histogram Matching**: Aligns intensity distributions between images

## Quick Start

### Installation in 3 Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/context-based-pseudo-coloring.git
cd context-based-pseudo-coloring
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run the Application
```bash
python final.py
```

### First Time Usage

1. **Launch the application** by running `python final.py`
2. **Select your input image** (grayscale) using "Select Input Image" button
3. **Select your reference image** (color) using "Select Reference Image" button
4. **Choose a coloring method:**
   - **Local Pseudocolor (Same Region)**: For coloring matching regions
   - **Global Pseudocolor**: For full image coloring with multiple techniques
   - **Local Pseudocolor (Different Region)**: For coloring different regions

### Example Workflow

#### Global Pseudocoloring
```bash
# Run the GUI
python final.py

# Steps in GUI:
# 1. Select Input Image → Choose grayscale image
# 2. Select Reference Image → Choose color image
# 3. Click "Global Pseudocolor"
# 4. View results in multiple windows
```

#### Local Pseudocoloring (Same Region)
```bash
# Run the GUI
python final.py

# Steps in GUI:
# 1. Select Input Image → Choose grayscale image
# 2. Select Reference Image → Choose color image
# 3. Click "Local Pseudocolor (Same Region)"
# 4. Draw ROI on input image (left-click and drag)
# 5. Press 'q' to confirm
# 6. View colorized result
```

#### Local Pseudocoloring (Different Regions)
```bash
# Run the GUI
python final.py

# Steps in GUI:
# 1. Select Input Image → Choose grayscale image
# 2. Select Reference Image → Choose color image
# 3. Click "Local Pseudocolor (Different Region)"
# 4. Draw ROI on input image → Press 'q'
# 5. Draw ROI on reference image → Press 'q'
# 6. View colorized result
```

### Tips for Best Results

1. **Image Quality**: Use high-quality images for better results
2. **Similar Content**: Reference and input images should have similar objects/scenes
3. **Resolution**: Images are automatically resized to 512x512 for processing
4. **ROI Selection**: Draw ROI carefully for accurate local coloring
5. **Processing Time**: 
   - Global histogram matching: Fast
   - Object-based matching: Moderate
   - Haralick features: Slow (avoid for quick tests)

### Keyboard Shortcuts

#### During ROI Selection:
- **Left-click + Drag**: Draw region of interest
- **Q**: Confirm selection and proceed
- **R**: Reset selection and start over

## Requirements

### System Requirements
- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: ~100MB for dependencies

### Python Dependencies

```
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0
numpy>=1.19.0
```

These dependencies are listed in `requirements.txt` and can be installed with:
```bash
pip install -r requirements.txt
```

### Package Descriptions

- **opencv-python**: Core OpenCV library for image processing operations
- **opencv-contrib-python**: Additional OpenCV modules with extra functionality
- **numpy**: Numerical computing library for array operations
- **tkinter**: GUI toolkit (usually comes pre-installed with Python)

## ✨ Features

### 1. Global Pseudocoloring
- **Direct Color Mapping**: Maps grayscale values to colors using a colormap derived from the reference image
- **Object-Based Matching**: Detects objects in both images and matches them based on geometric and textural features
- **Histogram Matching**: Adjusts the global histogram of the input image to match the reference image
- **Feature-Based Matching**: Uses area, perimeter, Hu moments, and other shape descriptors
- **Haralick Features**: Employs texture-based features (energy, contrast, homogeneity) for matching

### 2. Local Pseudocoloring
- **Same Region Coloring**: Colors the same region in both input and reference images
- **Different Region Coloring**: Allows coloring different regions between images
- **Interactive ROI Selection**: User-friendly interface for selecting regions of interest
- **Local Histogram Matching**: Matches histograms within selected regions

### 3. User Interface
- **Tkinter-Based GUI**: Easy-to-use graphical interface
- **Image Preview**: Display input and reference images before processing
- **Multiple Processing Options**: Choose from various pseudocoloring techniques
- **Scrollable Interface**: Accommodates multiple result windows

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/context-based-pseudo-coloring.git
cd context-based-pseudo-coloring
```

### Step 2: Create a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Alternative: Manual Installation**
```bash
pip install opencv-python opencv-contrib-python numpy
```

**Note**: Tkinter usually comes pre-installed with Python. If not:
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **Fedora**: `sudo dnf install python3-tkinter`
- **macOS**: Pre-installed with Python

### Step 4: Verify Installation

```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import tkinter; print('Tkinter: OK')"
```

## Usage

### Running the Main Application

The main application provides a GUI for easy interaction:

```bash
python final.py
```

**Steps to use:**
1. Click "Select Input Image" to choose your grayscale image
2. Click "Select Reference Image" to choose your color reference image
3. Choose from the available pseudocoloring options:
   - **Local Pseudocolor (Same Region)**: Color matching regions
   - **Global Pseudocolor**: Color using global techniques
   - **Local Pseudocolor (Different Region)**: Color different regions

### Running Haralick Feature-Based Matching

For texture-based object matching (note: computationally intensive):

```bash
python Haralick.py
```

**Note:** You'll need to modify the image paths in the script:
```python
input_image = cv2.imread(r"path/to/your/input.jpg", cv2.IMREAD_GRAYSCALE)
reference_image = cv2.imread(r"path/to/your/reference.jpg", cv2.IMREAD_GRAYSCALE)
reference_color_image = cv2.imread(r"path/to/your/reference.jpg")
```

### Running Individual Modules

**Global Pseudocoloring:**
```bash
python Global_Pseudo.py
```

**Object-Based Matching:**
```bash
python glob_obj.py
```

**Local Histogram Matching:**
```bash
python Local_hist.py
```

**ROI-Different Regions:**
```bash
python roi_diff.py
```

## Methodology

### Global Pseudocoloring Workflow

1. **Preprocessing**: Apply morphological closing to fill gaps in objects
2. **Contour Detection**: Extract object boundaries using edge detection
3. **Feature Extraction**: Calculate feature vectors for each object:
   - Area, perimeter, roundness, form factor
   - Axis ratio, orientation, extent, solidity
   - Hu moments (7 invariant moments)
   - Haralick features (optional, for texture analysis)
4. **Object Matching**: Match objects using Euclidean distance between feature vectors
5. **Color Mapping**: Create colormap from reference image and apply to matched objects
6. **Histogram Matching** (optional): Align intensity distributions before coloring

### Local Pseudocoloring Workflow

1. **ROI Selection**: User interactively selects regions of interest
   - Left-click and drag to draw ROI
   - Press 'q' to confirm selection
   - Press 'r' to reset selection
2. **Colormap Generation**: Create colormap from reference ROI
3. **Histogram Matching**: Match histograms between input and reference ROIs
4. **Selective Coloring**: Apply colors only to selected regions

### Feature Descriptors

**Geometric Features:**
- **Area (A)**: Total number of pixels in the object
- **Perimeter (P)**: Length of the object boundary
- **Roundness**: `4πA / P²`
- **Form Factor**: `4πA / P²`
- **Axis Ratio**: Major axis / Minor axis
- **Orientation**: Angle of the major axis
- **Extent**: Area / Bounding box area
- **Solidity**: Area / Convex hull area
- **Eccentricity**: `√(1 - (minor axis)² / (major axis)²)`

**Texture Features (Haralick):**
- **Energy**: `Σᵢ Σⱼ P(i,j)²`
- **Contrast**: Measures local intensity variation
- **Homogeneity**: Measures similarity of neighboring pixels

## Project Structure

```
context-based-pseudo-coloring/
│
├── final.py                    # Main GUI application
├── Global_Pseudo.py           # Global pseudocoloring methods
├── glob_obj.py                # Object-based matching and coloring
├── Haralick.py                # Haralick feature-based matching
├── Local_hist.py              # Local histogram matching (same region of 2 images)
├── roi_diff.py                # Local ROI matching (different regions of 2 images)
├── Report.pdf                 # Detailed project report
├── Presentation.pptx          # Project Presentation
├── .gitignore                 # Git ignore rules
├── output_Sample.zip          # Sample output
├── Test Image (Sample).zip    # Sample Test Images
└── README.md                  # This file
```

## Technical Details

### Color Mapping Algorithm

The colormap is created by averaging RGB values for each grayscale intensity:

```python
for each pixel in reference image:
    gray_value = grayscale(pixel)
    accumulate RGB values for gray_value
    
for each intensity level (0-255):
    colormap[intensity] = mean(RGB values)
```

### Histogram Matching Process

1. Calculate cumulative distribution function (CDF) for input and reference
2. Create lookup table by matching CDF values
3. Transform input image using the lookup table

### Object Matching

Objects are matched using minimum Euclidean distance:

```
distance = √(Σ(feature_input[i] - feature_reference[i])²)
```

## Limitations

1. **Object Matching Accuracy**: Feature-based matching may fail with significant size/orientation differences
2. **Computational Intensity**: Haralick feature computation is time-consuming
3. **Orientation Sensitivity**: Limited handling of rotated objects
4. **Color Accuracy**: Average local colormap approach trades accuracy for efficiency
5. **Morphological Operations**: May not fully eliminate all noise or fill all gaps

## Troubleshooting

### Common Issues

#### 1. OpenCV Import Error
```
ModuleNotFoundError: No module named 'cv2'
```
**Solution:**
```bash
pip install opencv-python opencv-contrib-python
```

#### 2. tkinter Not Found
```
ModuleNotFoundError: No module named 'tkinter'
```
**Solution:**
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **Fedora**: `sudo dnf install python3-tkinter`
- **macOS**: tkinter comes pre-installed with Python

#### 3. Image Not Displaying
**Possible Causes:**
- Unsupported image format
- Incorrect file path
- Corrupted image file

**Solution:**
- Ensure images are in supported formats (JPG, PNG, BMP, GIF)
- Check file paths are correct
- Try opening the image in another program to verify it's not corrupted

#### 4. Slow Performance
**Solution:**
- Reduce image size (images are resized to 512x512 internally)
- Avoid Haralick features for quick processing
- Use global histogram matching instead of object-based matching
- Close other applications to free up RAM

#### 5. GUI Not Responding
**Solution:**
- Wait for processing to complete (check console for progress)
- Restart the application
- Check system resources (RAM, CPU usage)


# Image-Pseudo-color
