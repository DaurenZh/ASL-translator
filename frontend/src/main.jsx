import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Camera, Copy, ImageUp, Pause, Play, RotateCcw, Trash2, Volume2 } from "lucide-react";
import "./styles.css";

const API_URL = "http://localhost:8000";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileRef = useRef(null);
  const streamRef = useRef(null);
  const lastAcceptedRef = useRef("");
  const [running, setRunning] = useState(false);
  const [sentence, setSentence] = useState("");
  const [current, setCurrent] = useState({ label: "-", confidence: 0, predictions: [] });
  const [threshold, setThreshold] = useState(0.85);
  const [voice, setVoice] = useState(false);
  const [status, setStatus] = useState("idle");
  const [uploadedImage, setUploadedImage] = useState("");
  const [uploadResult, setUploadResult] = useState(null);

  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    streamRef.current = stream;
    videoRef.current.srcObject = stream;
    setRunning(true);
    setStatus("camera");
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setRunning(false);
    setStatus("paused");
  }

  function applyToken(label) {
    if (!label || label === "nothing" || label === lastAcceptedRef.current) return;
    lastAcceptedRef.current = label;

    setSentence((value) => {
      if (label === "space") return `${value} `;
      if (label === "del") return value.slice(0, -1);
      return `${value}${label}`;
    });
  }

  async function predictFrame() {
    if (!videoRef.current || !canvasRef.current || !running) return;
    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    const image = canvas.toDataURL("image/jpeg", 0.8);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image }),
      });
      const result = await response.json();
      setCurrent(result);
      setStatus(result.ready ? "live" : "model missing");
      if (result.confidence >= threshold) applyToken(result.label);
      if (result.confidence < threshold) lastAcceptedRef.current = "";
    } catch {
      setStatus("api offline");
    }
  }

  async function predictImageData(image) {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    return response.json();
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async () => {
      const image = reader.result;
      setUploadedImage(image);
      setStatus("uploading");
      try {
        const result = await predictImageData(image);
        setUploadResult(result);
        setCurrent(result);
        setStatus(result.ready ? "upload result" : "model missing");
      } catch {
        setStatus("api offline");
      }
    };
    reader.readAsDataURL(file);
  }

  useEffect(() => {
    if (!running) return undefined;
    const timer = setInterval(predictFrame, 650);
    return () => clearInterval(timer);
  }, [running, threshold]);

  function speak() {
    if (!sentence.trim()) return;
    const utterance = new SpeechSynthesisUtterance(sentence);
    window.speechSynthesis.speak(utterance);
  }

  useEffect(() => {
    if (voice && sentence.trim()) speak();
  }, [voice]);

  return (
    <main className="shell">
      <section className="cameraPane">
        <video ref={videoRef} autoPlay playsInline muted />
        <canvas ref={canvasRef} hidden />
        <div className="statusBar">
          <span>{status}</span>
          <strong>{current.label}</strong>
          <span>{Math.round((current.confidence || 0) * 100)}%</span>
        </div>
      </section>

      <aside className="panel">
        <div className="brand">
          <Camera size={24} />
          <h1>ASL Translator</h1>
        </div>

        <div className="prediction">
          <span>Current sign</span>
          <strong>{current.label}</strong>
          <meter value={current.confidence || 0} min="0" max="1" />
        </div>

        <div className="sentence">{sentence || "..."}</div>

        <div className="toolbar">
          <button title={running ? "Pause camera" : "Start camera"} onClick={running ? stopCamera : startCamera}>
            {running ? <Pause /> : <Play />}
          </button>
          <button title="Clear sentence" onClick={() => setSentence("")}>
            <Trash2 />
          </button>
          <button title="Copy sentence" onClick={() => navigator.clipboard.writeText(sentence)}>
            <Copy />
          </button>
          <button title="Speak sentence" onClick={speak}>
            <Volume2 />
          </button>
          <button title="Reset current sign" onClick={() => { lastAcceptedRef.current = ""; setCurrent({ label: "-", confidence: 0, predictions: [] }); }}>
            <RotateCcw />
          </button>
          <button title="Upload image" onClick={() => fileRef.current?.click()}>
            <ImageUp />
          </button>
        </div>

        <input ref={fileRef} className="fileInput" type="file" accept="image/*" onChange={handleUpload} />

        <section className="uploadBox">
          {uploadedImage ? <img src={uploadedImage} alt="Uploaded hand sign" /> : <div className="emptyUpload">Upload sample</div>}
          <div>
            <span>Uploaded result</span>
            <strong>{uploadResult?.label || "-"}</strong>
            <p>{Math.round((uploadResult?.confidence || 0) * 100)}%</p>
          </div>
        </section>

        <label className="slider">
          <span>Sensitivity {Math.round(threshold * 100)}%</span>
          <input
            type="range"
            min="0.5"
            max="0.98"
            step="0.01"
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value))}
          />
        </label>

        <label className="toggle">
          <input type="checkbox" checked={voice} onChange={(event) => setVoice(event.target.checked)} />
          <span>Voice</span>
        </label>

        <div className="topList">
          {(current.predictions || []).map((item) => (
            <div key={item.label}>
              <span>{item.label}</span>
              <strong>{Math.round(item.confidence * 100)}%</strong>
            </div>
          ))}
        </div>
      </aside>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
