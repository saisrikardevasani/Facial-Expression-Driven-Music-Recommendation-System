# Facial Expression-Driven Music Recommendation System: Comprehensive Project Report

**Author:** Sai Srikar Devasani  
**Date:** February 2026  
**Repository:** [github.com/saisrikardevasani/Facial-Expression-Driven-Music-Recommendation-System](https://github.com/saisrikardevasani/Facial-Expression-Driven-Music-Recommendation-System)  
**Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/srikarcod3r/Facial_Expression_Driven_Music_Recommendation_System)

---

## 1. Project Overview

### 1.1 Objectives
The primary objective of this project is to develop an end-to-end system that:
1. **Detects facial expressions** from static images or live webcam feeds using a custom Convolutional Neural Network (CNN).
2. **Classifies emotions** into one of seven categories: angry, disgust, fear, happy, neutral, sad, and surprise.
3. **Recommends music** by mapping the detected emotion to a curated Spotify playlist, providing a personalized listening experience based on the user's emotional state.

### 1.2 Key Milestones
| Milestone | Description | Status |
|---|---|---|
| Data Acquisition | Sourcing and organizing the FER-2013 dataset | ✅ Complete |
| Model Design | Custom Sequential CNN architecture with 4 convolutional blocks | ✅ Complete |
| Model Training | 100-epoch training with Adam optimizer | ✅ Complete |
| Model Export | Serialization of architecture (JSON) and weights (.h5/.keras) | ✅ Complete |
| Local Detection | Real-time webcam emotion detection via OpenCV | ✅ Complete |
| Web Deployment | Gradio interface on Hugging Face Spaces | ✅ Complete |
| Music Integration | Spotify playlist mapping for all 7 emotion classes | ✅ Complete |
| Research Paper | Published academic paper on the methodology | ✅ Complete |
| Documentation | Comprehensive README, report, and repository organization | ✅ Complete |

### 1.3 Deliverables
- Jupyter notebook (`Trained model.ipynb`) with the complete training pipeline and outputs.
- Serialized model architecture (`emotiondetector.json`) and trained weights (`emotiondetector.h5`).
- Real-time detection scripts (`realtimedetection.py`, `webcampermission.py`).
- Gradio web application (`app.py`) deployed on Hugging Face Spaces.
- Published research paper (`Facial_Expression-Driven_Music_Recommendation_Paper.pdf`).
- Project report (this document) and README documentation.

---

## 2. Data Collection and Preparation

### 2.1 Data Source
The dataset is based on the **FER-2013 (Facial Expression Recognition 2013)** benchmark, one of the most widely used datasets for facial emotion classification research. The images are organized in a directory structure by emotion class.

**Dataset Statistics:**
| Split | Samples |
|---|---|
| Training set | **28,709** |
| Test set | **7,178** |
| **Total** | **35,887** |

**Image Specifications:**
- Resolution: 48 × 48 pixels
- Color mode: Grayscale (single channel)
- Format: Individual image files organized by emotion label

### 2.2 Emotion Class Distribution
| Class ID | Emotion | Description |
|---|---|---|
| 0 | Angry | Furrowed brows, tense jaw, narrowed eyes |
| 1 | Disgust | Wrinkled nose, raised upper lip |
| 2 | Fear | Wide eyes, raised eyebrows, open mouth |
| 3 | Happy | Smile, raised cheeks, crow's feet around eyes |
| 4 | Neutral | Relaxed facial muscles, no strong expression |
| 5 | Sad | Downturned mouth, drooping eyes |
| 6 | Surprise | Raised eyebrows, wide-open eyes and mouth |

### 2.3 Data Preprocessing Pipeline
1. **Image Loading:** Images loaded using `keras_preprocessing.image.load_img` in grayscale mode at native 48×48 resolution.
2. **Array Conversion:** Each image converted to a NumPy array and reshaped to `(48, 48, 1)` to include the channel dimension.
3. **Normalization:** Pixel intensity values scaled from [0, 255] to [0.0, 1.0] via division by 255.0.
4. **Label Encoding:** String labels encoded to integers using `sklearn.preprocessing.LabelEncoder` (alphabetical order: angry=0, disgust=1, ..., surprise=6).
5. **One-Hot Encoding:** Integer labels converted to 7-dimensional one-hot vectors using `keras.utils.to_categorical`.

---

## 3. Model Architecture

### 3.1 Network Design
A custom **Sequential CNN** architecture was designed with progressively increasing filter depths to capture hierarchical facial features:

| # | Layer | Configuration | Output Shape |
|---|---|---|---|
| 1 | Input | shape=(48, 48, 1) | (48, 48, 1) |
| 2 | Conv2D | 128 filters, 3×3, ReLU | (46, 46, 128) |
| 3 | MaxPooling2D | pool 2×2 | (23, 23, 128) |
| 4 | Dropout | rate=0.4 | (23, 23, 128) |
| 5 | Conv2D | 256 filters, 3×3, ReLU | (21, 21, 256) |
| 6 | MaxPooling2D | pool 2×2 | (10, 10, 256) |
| 7 | Dropout | rate=0.4 | (10, 10, 256) |
| 8 | Conv2D | 512 filters, 3×3, ReLU | (8, 8, 512) |
| 9 | MaxPooling2D | pool 2×2 | (4, 4, 512) |
| 10 | Dropout | rate=0.4 | (4, 4, 512) |
| 11 | Conv2D | 512 filters, 3×3, ReLU | (2, 2, 512) |
| 12 | MaxPooling2D | pool 2×2 | (1, 1, 512) |
| 13 | Dropout | rate=0.4 | (1, 1, 512) |
| 14 | Flatten | — | (512,) |
| 15 | Dense | 512 units, ReLU | (512,) |
| 16 | Dropout | rate=0.4 | (512,) |
| 17 | Dense | 256 units, ReLU | (256,) |
| 18 | Dropout | rate=0.3 | (256,) |
| 19 | Dense (Output) | 7 units, Softmax | (7,) |

### 3.2 Design Rationale
- **Progressive filter depth (128 → 256 → 512 → 512):** Lower layers capture low-level features (edges, textures); deeper layers capture high-level semantic features (facial expressions).
- **Aggressive Dropout (0.3–0.4):** Applied after every pooling layer and dense layer to combat overfitting, which is critical given the relatively small and noisy FER-2013 dataset.
- **No Batch Normalization:** Simplifies the architecture while relying on dropout for regularization.
- **Softmax output:** Produces a probability distribution over 7 emotion classes, enabling confidence-based predictions.

---

## 4. Model Training and Tuning

### 4.1 Training Configuration
| Parameter | Value |
|---|---|
| Optimizer | Adam (default learning rate) |
| Loss Function | Categorical Crossentropy |
| Metrics | Accuracy |
| Batch Size | 128 |
| Epochs | 100 |
| Validation Data | Test set (7,178 samples) |

### 4.2 Training Progression
The model was trained for 100 epochs. Key milestones from the training log:

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|---|---|---|---|---|
| 1 | 23.6% | 1.840 | 24.7% | 1.817 |
| 10 | 48.7% | 1.357 | 51.8% | 1.250 |
| 25 | 56.0% | 1.160 | 57.7% | 1.109 |
| 50 | 61.7% | 1.019 | 60.7% | 1.044 |
| 75 | 66.7% | 0.891 | 62.5% | 1.025 |
| 100 | 71.1% | 0.789 | 62.9% | 1.025 |

### 4.3 Training Observations
1. **Rapid early convergence:** Accuracy increased from ~24% to ~52% within the first 10 epochs, indicating the model quickly learned basic facial features.
2. **Gradual improvement:** From epochs 10–50, validation accuracy steadily improved from ~52% to ~61%.
3. **Plateau region:** After epoch 50, validation accuracy oscillated between 60–63%, while training accuracy continued to improve, indicating the onset of overfitting.
4. **Overfitting gap:** By epoch 100, the training-validation accuracy gap was approximately 8 percentage points (71.1% vs 62.9%), a moderate level of overfitting controlled by the aggressive dropout scheme.

---

## 5. Performance Analysis

### 5.1 Final Model Performance
| Metric | Training Set | Test Set |
|---|---|---|
| **Accuracy** | 71.1% | 62.9% |
| **Loss** | 0.789 | 1.025 |

### 5.2 Contextual Benchmarking
The FER-2013 dataset is known to be a challenging benchmark due to:
- **Label noise:** Estimated ~10–15% of images are ambiguously or incorrectly labeled.
- **Low resolution:** 48×48 grayscale images lose significant facial detail.
- **Class imbalance:** Some emotions (e.g., disgust) are significantly underrepresented.
- **Inter-class similarity:** Emotions like fear and surprise share similar facial features.

For context, published results on FER-2013:
| Method | Accuracy |
|---|---|
| Human-level performance | ~65% ± 5% |
| Simple CNN baselines | 60–65% |
| VGG-based models | 70–73% |
| State-of-the-art (ensemble/attention) | 74–76% |

Our model's **62.9% validation accuracy** is competitive with simple CNN baselines and approaches human-level performance on this dataset.

### 5.3 Model Verification
The notebook includes test predictions demonstrating correct classification:
- **Input:** Training sample labeled "angry" → **Prediction:** angry ✅
- **Input:** Test sample labeled "happy" → **Prediction:** happy ✅

---

## 6. Deployment

### 6.1 Gradio Web Interface (Hugging Face Spaces)
The model is deployed as an interactive web application using Gradio on Hugging Face Spaces.

**Application Architecture:**
```
User Input (Image) → Preprocessing → CNN Prediction → Emotion Label → Spotify Playlist URL
                                                     → Random Fun Fact Display
```

**Key Features:**
- Accept images via upload, webcam capture, or paste
- Display detected emotion label
- Show a random fun fact about emotions and music
- Provide a direct link to a curated Spotify playlist matching the detected mood

### 6.2 Spotify Playlist Mapping
| Detected Emotion | Playlist Style | Platform |
|---|---|---|
| Happy | Happy Hits | Spotify |
| Sad | Sad Songs | Spotify |
| Angry | Intense / Heavy | Spotify |
| Fear | Ambient / Chill | Spotify |
| Neutral | Chill Vibes | Spotify |
| Surprise | Upbeat / Party | Spotify |
| Disgust | Cathartic / Release | Spotify |

### 6.3 Real-Time Webcam Detection
Two local scripts provide real-time emotion detection:

**`realtimedetection.py` / `webcampermission.py`:**
1. Load the trained model from `emotiondetector.json` + `emotiondetector.h5`
2. Initialize webcam via OpenCV's `VideoCapture`
3. Detect faces using Haar Cascade classifier (`haarcascade_frontalface_default.xml`)
4. For each detected face: crop, resize to 48×48, normalize, and predict emotion
5. Annotate the live video feed with bounding boxes and predicted emotion labels

---

## 7. Documentation and Reporting

### 7.1 Documentation Practices
- **Code Comments:** In-line comments throughout the Jupyter notebook explain each processing step and design choice.
- **README.md:** Provides a concise project overview, quick-start instructions, performance summary, and repository structure.
- **Project Report (this document):** Formal, comprehensive analysis covering all phases of the ML pipeline.
- **Research Paper:** Published academic paper detailing the methodology and results.

### 7.2 Repository Structure
```
Facial-Expression-Driven-Music-Recommendation-System/
├── README.md                                            # Project documentation
├── Project_Report.md                                    # This comprehensive report
├── Trained model.ipynb                                  # CNN training notebook
├── emotiondetector.json                                 # Model architecture
├── emotiondetection.py                                  # Basic webcam capture
├── realtimedetection.py                                 # Real-time emotion detection
├── webcampermission.py                                  # Alternative webcam detection
├── requirement.txt                                      # Python dependencies
├── Facial_Expression-Driven_Music_Recommendation_Paper.pdf  # Published paper
└── LICENSE                                              # MIT License
```

### 7.3 External Deployment
- **Hugging Face Spaces:** [Live application](https://huggingface.co/spaces/srikarcod3r/Facial_Expression_Driven_Music_Recommendation_System) with model weights and Gradio `app.py`

---

## 8. Conclusion

### 8.1 Project Outcomes
This project successfully demonstrates the end-to-end development and deployment of a facial emotion recognition system with practical music recommendation capabilities:

- A custom **4-block CNN** achieved **62.9% validation accuracy** on the FER-2013 benchmark, competitive with simple CNN baselines and approaching human-level performance on this notoriously challenging dataset.
- The system is deployed as a **live web application** on Hugging Face Spaces, providing instant emotion detection and Spotify playlist recommendations.
- Real-time webcam detection scripts enable local, offline usage of the trained model.

### 8.2 Lessons Learned
1. **FER-2013 is inherently noisy:** The ~10–15% label noise rate in FER-2013 fundamentally limits achievable accuracy. Data cleaning or label smoothing could improve results.
2. **Dropout is essential for small datasets:** The aggressive dropout scheme (0.3–0.4) effectively contained overfitting, keeping the train-val gap to ~8 percentage points despite 100 training epochs.
3. **Gradio simplifies deployment:** The Gradio framework enabled rapid prototyping and deployment of an interactive ML application with minimal frontend code.
4. **Emotion-to-music mapping enhances UX:** Connecting facial expression detection to actionable music recommendations creates a tangible, engaging user experience beyond raw classification.

### 8.3 Recommendations for Future Work
- **Architecture Improvements:** Evaluate transfer learning with pre-trained models (VGG-16, ResNet-50, EfficientNet) for improved accuracy.
- **Data Augmentation:** Apply random rotation, flipping, brightness adjustment, and elastic deformation to increase effective training set size.
- **Class Imbalance Handling:** Implement class-weighted loss or oversampling strategies for underrepresented emotions (especially disgust).
- **Ensemble Methods:** Combine predictions from multiple models to improve robustness and accuracy.
- **Advanced Music Recommendation:** Integrate the Spotify API for dynamic playlist generation based on emotion confidence scores rather than hard label mapping.
- **Multi-Face Detection:** Extend the system to handle multiple faces in a single image, detecting and recommending music based on the dominant group emotion.

---

*End of Report*
