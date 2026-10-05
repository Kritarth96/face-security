# 🚀 Real-Time Face Recognition System

A production-oriented **real-time face recognition system** built using **SCRFD** for face detection and **ArcFace** for facial recognition. The system performs high-speed recognition using embedding-based matching and supports enrollment of multiple users with real-time webcam inference.

---

## ✨ Features

- 🔍 Real-time face detection using **SCRFD**
- 👤 Face recognition using **ArcFace embeddings**
- 📸 Webcam-based live recognition
- ➕ User enrollment pipeline
- ❌ Unknown face rejection using configurable threshold
- 💾 Embedding database stored locally
- ⚡ Background-threaded recognition for smooth UI
- 📊 Runtime performance metrics (FPS, CPU, RAM, Recognition Time)
- 🧩 Modular project architecture

---

## 📂 Project Structure

```
face-recognition-poc/
│
├── src/
│   ├── detector/
│   ├── embeddings/
│   ├── recognition/
│   ├── benchmark/
│   ├── utils/
│   ├── main.py
│   └── enroll.py
│
├── embeddings/
├── reports/
├── weights/
│
├── README.md
└── requirements.txt
```

---

## 🏗️ System Architecture

```
Reference Image
       │
       ▼
Enrollment
       │
       ▼
SCRFD Face Detection
       │
       ▼
Face Crop
       │
       ▼
ArcFace Embedding
       │
       ▼
Embedding Database
       ▲
       │
       │
Webcam
       │
       ▼
SCRFD Detection
       │
       ▼
Face Crop
       │
       ▼
ArcFace Embedding
       │
       ▼
Cosine Similarity
       │
       ▼
Recognized Person / Unknown
```

---

## 🛠️ Tech Stack

- Python 3.11
- OpenCV
- DeepFace
- InsightFace
- SCRFD
- ArcFace
- NumPy
- Pickle
- PSUtil

---

## 📈 Performance

| Metric | Value |
|---------|-------|
| Face Detector | SCRFD |
| Recognition Model | ArcFace |
| Average FPS | **40–45 FPS** |
| Average CPU Usage | **~30%** |
| Average RAM Usage | **~1200 MB** |
| Stable Face Accuracy | **~95%** |
| Mixed Pose Accuracy | **~75%** |

> Mixed pose testing includes head tilts, side poses and rotations greater than approximately 45°.

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/face-recognition-poc.git
```

Move inside the project

```bash
cd face-recognition-poc
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📸 Enrolling a New Person

Run

```bash
python src/enroll.py
```

Enter

```
Person Name
Reference Image Path
```

The system will

- Detect face
- Crop face
- Generate ArcFace embedding
- Store embedding in local database

---

## ▶️ Running Recognition

```bash
python src/main.py
```

The webcam window displays

- Recognized Person
- Distance Score
- FPS
- CPU Usage
- RAM Usage
- Recognition Time

---

## 📁 Embedding Database

The system stores face embeddings locally using Pickle.

```
embeddings/
    embeddings.pkl
```

Each identity is represented by a **512-dimensional ArcFace embedding**.

---

## 🎯 Design Decisions

### Why SCRFD instead of YOLO?

Although YOLO is an excellent general-purpose object detector, **SCRFD is specifically designed for face detection** and provides:

- Higher face detection accuracy
- Better localization
- Facial landmarks
- Faster face inference

making it better suited for facial recognition pipelines.

---

### Why ArcFace?

ArcFace provides highly discriminative facial embeddings, making recognition more accurate while generating compact **512-dimensional feature vectors**.

---

## 📌 Current Capabilities

- ✅ Single-face recognition
- ✅ Unknown face rejection
- ✅ Multiple user enrollment
- ✅ Real-time webcam inference
- ✅ CPU-only execution

---

## 🚧 Future Improvements

- Multi-face recognition
- FastAPI REST API
- Docker deployment
- SQLite / PostgreSQL embedding storage
- Raspberry Pi optimization
- Face recognition SDK

---

## 👨‍💻 Author

**Kritarth Upadhyay**

---

## 📄 License

This project is licensed under the MIT License.
