# AI Face Liveness Detection

A real-time face liveness detection system built with **React**, **FastAPI**, **OpenCV**, **MediaPipe**, and **ONNX Runtime**.

The app verifies whether a user is a real live person by combining face landmark detection, blink/head movement challenges, and an anti-spoofing model.

## Live Demo

https://ai-face-liveness-react-fastapi-n3lfgmd0f-ar040701s-projects.vercel.app/

## Features

- Real-time webcam-based verification
- Random liveness challenges:
  - Blink
  - Turn left
  - Turn right
- Face landmark detection with MediaPipe
- Blink detection using Eye Aspect Ratio
- Head movement detection
- Anti-spoofing using ONNX Runtime
- React frontend and FastAPI backend

## Tech Stack

| Part | Technology |
|---|---|
| Frontend | React, Vite |
| Backend | FastAPI |
| Computer Vision | OpenCV, MediaPipe |
| Model Inference | ONNX Runtime |
| Deployment | Vercel, Render |

## Project Structure

```text
ai-face-liveness-react-fastapi/
│
├── backend/
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── config/
│   ├── modules/
│   └── models/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

## How It Works

```text
Webcam frame
    ↓
React captures image
    ↓
Frame is sent to FastAPI
    ↓
Backend runs face detection, challenge checks, and anti-spoofing
    ↓
Result is returned to React
```

The user is marked as **LIVE** only when:

```python
anti_spoof_decision == "REAL" and challenge_ok
```

## Status Values

| Status | Meaning |
|---|---|
| `CHECKING` | Verification is in progress |
| `CHALLENGE_WARNING` | User is taking too long |
| `TIMEOUT` | Challenge was not completed |
| `LIVE` | User passed verification |
| `SPOOF` | Fake face detected |
| `NO_FACE` | No face detected |
| `BAD_IMAGE` | Poor image quality |
| `ANTI_SPOOF_ERROR` | Model missing or failed |

## Local Setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

## Environment Variable

For production frontend deployment, set:

```env
VITE_API_BASE_URL=https://your-render-backend-url.onrender.com
```

## Deployment

- **Frontend:** Vercel
- **Backend:** Render with Docker

## Model Files

The backend expects:

```text
backend/models/best_model.onnx
backend/models/face_landmarker.task
```

Model paths are configured in:

```text
backend/config/settings.py
```

## Note

This project is for educational and demonstration purposes. It is not a certified biometric security system.
