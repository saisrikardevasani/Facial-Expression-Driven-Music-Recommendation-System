# Facial Expression-Driven Music Recommendation System

## Project Overview

This project implements a **deep learning-based facial emotion recognition system** that detects human facial expressions from images and recommends Spotify playlists tailored to the detected mood. The system combines a custom **Convolutional Neural Network (CNN)** trained on the FER-2013 dataset with a **Gradio web interface** deployed on Hugging Face Spaces.

### Key Objectives
- Train a CNN model to classify facial expressions into **7 emotion categories**: angry, disgust, fear, happy, neutral, sad, and surprise.
- Deploy an interactive web application where users can upload or capture facial images and receive emotion-based music recommendations.
- Provide a real-time webcam-based emotion detection script for local use.

### Deliverables
| Deliverable | Description |
|---|---|
| `Trained model.ipynb` | Jupyter notebook with end-to-end model training pipeline |
| `emotiondetector.json` | Serialized CNN model architecture (JSON) |
| `emotiondetector.h5` | Trained model weights (hosted on Hugging Face) |
| `realtimedetection.py` | Real-time webcam emotion detection script |
| `webcampermission.py` | Alternative webcam detection implementation |
| `emotiondetection.py` | Basic webcam capture utility |
| `requirement.txt` | Python dependency listing |
| `Facial_Expression-Driven_Music_Recommendation_Paper.pdf` | Published research paper |
| `Project_Report.md` | Comprehensive project report |

---

## Live Demo

🚀 **Try the deployed application:** [Hugging Face Spaces — Facial Expression Driven Music Recommendation](https://huggingface.co/spaces/srikarcod3r/Facial_Expression_Driven_Music_Recommendation_System)

Upload a photo of a face and the app will:
1. Recognize the expressed emotion (e.g., happy, sad, angry)
2. Display a random fun fact about emotions
3. Provide a Spotify playlist that matches the detected mood

---

## Data Collection and Preparation

### Dataset
- **Source:** FER-2013 (Facial Expression Recognition 2013)
- **Image Format:** 48×48 pixel grayscale images
- **Training Set:** 28,709 images
- **Test Set:** 7,178 images
- **Total:** 35,887 images

### Emotion Classes
| Class | Label |
|---|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Neutral |
| 5 | Sad |
| 6 | Surprise |

### Preprocessing
- Images loaded in grayscale at 48×48 resolution
- Pixel values normalized to [0, 1] range (division by 255.0)
- Labels encoded using `sklearn.preprocessing.LabelEncoder`
- One-hot encoded to 7-class vectors via `keras.utils.to_categorical`

---

## Model Architecture

A custom **Sequential CNN** with 4 convolutional blocks followed by fully connected layers:

| Layer | Type | Parameters |
|---|---|---|
| Input | Input | Shape: (48, 48, 1) |
| Block 1 | Conv2D → MaxPool2D → Dropout | 128 filters, 3×3 kernel, ReLU, pool 2×2, dropout 0.4 |
| Block 2 | Conv2D → MaxPool2D → Dropout | 256 filters, 3×3 kernel, ReLU, pool 2×2, dropout 0.4 |
| Block 3 | Conv2D → MaxPool2D → Dropout | 512 filters, 3×3 kernel, ReLU, pool 2×2, dropout 0.4 |
| Block 4 | Conv2D → MaxPool2D → Dropout | 512 filters, 3×3 kernel, ReLU, pool 2×2, dropout 0.4 |
| Flatten | Flatten | — |
| Dense 1 | Dense → Dropout | 512 units, ReLU, dropout 0.4 |
| Dense 2 | Dense → Dropout | 256 units, ReLU, dropout 0.3 |
| Output | Dense | 7 units, Softmax |

**Total Parameters:** ~6.8M

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Batch Size | 128 |
| Epochs | 100 |
| Validation Data | Test set (7,178 images) |

---

## Performance Summary

| Metric | Training | Validation |
|---|---|---|
| **Accuracy** | ~71.1% | ~62.9% |
| **Loss** | ~0.789 | ~1.025 |

The model shows typical behavior for emotion recognition on FER-2013, where state-of-the-art CNN models typically achieve 65–75% accuracy on this challenging benchmark. The gap between training and validation accuracy indicates moderate overfitting, which is expected given the dataset's inherent noise (many FER-2013 images have ambiguous or mislabeled expressions).

---

## Deployment

### Gradio Web Interface (Hugging Face Spaces)
The trained model is deployed as an interactive web application using **Gradio** on **Hugging Face Spaces**:
- Users can upload, capture, or paste facial images
- The system detects the dominant emotion and maps it to a curated Spotify playlist
- A random fun fact about emotions and music is displayed alongside results

### Spotify Playlist Mapping
| Emotion | Playlist Type |
|---|---|
| Happy | Happy Hits |
| Sad | Sad Songs |
| Angry | Intense / Heavy |
| Fear | Ambient / Chill |
| Neutral | Chill Vibes |
| Surprise | Upbeat / Party |
| Disgust | Cathartic / Release |

### Real-Time Detection (Local)
The `realtimedetection.py` script provides real-time webcam-based emotion detection using:
- **OpenCV** for webcam capture and face detection (Haar Cascade classifier)
- The trained CNN model for emotion classification
- Live annotation of detected emotions on the video feed

---

## Quick Start

```bash
# Clone the repository
git clone git@github.com:saisrikardevasani/Facial-Expression-Driven-Music-Recommendation-System.git
cd Facial-Expression-Driven-Music-Recommendation-System

# Install dependencies
pip install -r requirement.txt

# For real-time webcam detection (requires model weights from Hugging Face)
# Download emotiondetector.h5 from:
# https://huggingface.co/spaces/srikarcod3r/Facial_Expression_Driven_Music_Recommendation_System
python realtimedetection.py

# Or open the training notebook
jupyter notebook "Trained model.ipynb"
```

---

## Repository Structure
```
Facial-Expression-Driven-Music-Recommendation-System/
├── README.md                                            # This documentation
├── Project_Report.md                                    # Comprehensive project report
├── Trained model.ipynb                                  # CNN training notebook (with outputs)
├── emotiondetector.json                                 # Model architecture (JSON)
├── emotiondetection.py                                  # Basic webcam capture script
├── realtimedetection.py                                 # Real-time emotion detection script
├── webcampermission.py                                  # Alternative webcam detection
├── requirement.txt                                      # Python dependencies
├── Facial_Expression-Driven_Music_Recommendation_Paper.pdf  # Published research paper
└── LICENSE                                              # MIT License
```

### External Resources
- **Trained Model Weights:** [Hugging Face Spaces](https://huggingface.co/spaces/srikarcod3r/Facial_Expression_Driven_Music_Recommendation_System) (emotiondetector.h5, ~50 MB)
- **Live Demo:** [Hugging Face Spaces App](https://huggingface.co/spaces/srikarcod3r/Facial_Expression_Driven_Music_Recommendation_System)

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
