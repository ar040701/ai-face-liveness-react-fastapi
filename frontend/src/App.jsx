import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  const [cameraStarted, setCameraStarted] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      videoRef.current.srcObject = stream;
      setCameraStarted(true);
    } catch (error) {
      alert("Camera permission denied or camera not available.");
      console.error(error);
    }
  };

  const stopCamera = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    const stream = videoRef.current?.srcObject;

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    setCameraStarted(false);
  };

  const captureFrame = async () => {
    if (!videoRef.current || !canvasRef.current || loading) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) return;

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      try {
        setLoading(true);

        const response = await fetch(`${API_BASE_URL}/verify-frame`, {
           method: "POST",
           body: formData,
          });

        const data = await response.json();
        setResult(data);
      } catch (error) {
        console.error("Verification failed:", error);
      } finally {
        setLoading(false);
      }
    }, "image/jpeg", 0.8);
  };

  const startVerification = () => {
    if (!cameraStarted) {
      alert("Start camera first.");
      return;
    }

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    intervalRef.current = setInterval(() => {
      captureFrame();
    }, 300);
  };

  const resetVerification = async () => {
    try {
      await fetch(`${API_BASE_URL}/reset`, {
        method: "POST",
      });

      setResult(null);
    } catch (error) {
      console.error("Reset failed:", error);
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const status = result?.status || "NOT_STARTED";

  return (
    <div className="app">
      <h1>AI Face Liveness Detection</h1>
      <p>Follow the random challenge shown on screen.</p>

      <div className="video-card">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="video"
        />

        <canvas ref={canvasRef} style={{ display: "none" }} />

        <div className={`status ${status}`}>
          Status: {status}
        </div>
      </div>

      <div className="buttons">
        <button onClick={startCamera}>Start Camera</button>
        <button onClick={startVerification}>Start Verification</button>
        <button onClick={resetVerification}>Reset</button>
        <button onClick={stopCamera}>Stop Camera</button>
      </div>

      <div className="result-card">
        <h2>Verification Result</h2>

        <p><strong>Message:</strong> {result?.message || "-"}</p>
        <p><strong>Challenge:</strong> {result?.current_challenge_text || "-"}</p>
        <p><strong>Completed:</strong> {result?.completed_challenges?.length || 0} / {result?.challenges?.length || 0}</p>
        <p><strong>Blinks:</strong> {result?.blink_count ?? "-"}</p>
        <p><strong>Head Direction:</strong> {result?.head_direction || "-"}</p>
        <p><strong>Image Quality:</strong> {String(result?.quality_ok ?? "-")}</p>
        <p><strong>Quality Message:</strong> {result?.quality_message || "-"}</p>
        <p><strong>Anti Spoof OK:</strong> {String(result?.spoof_ok ?? "-")}</p>
        <p><strong>Anti Spoof Decision:</strong> {result?.anti_spoof_decision || "-"}</p>
        <p><strong>Real Score:</strong> {result?.real_score ?? "-"}</p>
        <p><strong>Fake Score:</strong> {result?.fake_score ?? "-"}</p>
        <p><strong>Time Left:</strong> {result?.challenge_time_left ?? "-"}</p>
        <p><strong>Warning:</strong> {String(result?.challenge_warning ?? "-")}</p>
        <p><strong>Severe Quality Fail:</strong> {String(result?.severe_quality_fail ?? "-")}</p>
      </div>
    </div>
  );
}

export default App;