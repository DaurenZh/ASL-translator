import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Camera, Copy, ImageUp, Pause, Play, RotateCcw, Trash2, Volume2 } from "lucide-react";
import "./styles.css";

const API_URL = "http://localhost:8000";
const EMPTY_RESULT = { label: "-", confidence: 0, predictions: [] };

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileRef = useRef(null);
  const streamRef = useRef(null);
  const lastAcceptedRef = useRef("");
  const [running, setRunning] = useState(false);
  const [sentence, setSentence] = useState("");
  const [current, setCurrent] = useState(EMPTY_RESULT);
  const [threshold, setThreshold] = useState(0.85);
  const [voice, setVoice] = useState(false);
  const [status, setStatus] = useState("idle");
  const [uploadedImage, setUploadedImage] = useState("");
  const [uploadResult, setUploadResult] = useState(null);

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setRunning(true);
      setStatus("camera");
    } catch {
      setStatus("camera blocked");
    }
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

    if (voice && label !== "space" && label !== "del") speakText(label);
  }

  function resetPredictions() {
    lastAcceptedRef.current = "";
    setCurrent(EMPTY_RESULT);
    setUploadResult(null);
    setUploadedImage("");
    setStatus(running ? "camera" : "idle");
    if (fileRef.current) fileRef.current.value = "";
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
        setCurrent(EMPTY_RESULT);
        setStatus(result.ready ? "upload result" : "model missing");
        if (voice && result.ready && result.confidence >= threshold) speakText(result.label);
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

  function speakText(text) {
    if (!text?.trim()) return;
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  }

  function speakCurrent() {
    const text = sentence.trim() || uploadResult?.label || current.label;
    if (text && text !== "-") speakText(text);
  }

  const liveConfidence = Math.round((current.confidence || 0) * 100);
  const uploadConfidence = Math.round((uploadResult?.confidence || 0) * 100);
  const uploadAccepted = Boolean(uploadResult?.ready && uploadResult.confidence >= threshold);
  const canSpeak = Boolean(sentence.trim() || uploadResult?.label || (current.label && current.label !== "-"));

  return (
    <main className="shell">
      <section className="cameraPane">
        <video ref={videoRef} autoPlay playsInline muted />
        <canvas ref={canvasRef} hidden />
        <div className="statusBar">
          <span>{status}</span>
          <strong>{running ? current.label : "-"}</strong>
          <span>{running ? `${liveConfidence}%` : "camera off"}</span>
        </div>
      </section>

      <aside className="panel">
        <div className="brand">
          <Camera size={24} />
          <h1>ASL Translator</h1>
        </div>

        <div className="prediction">
          <span>Live sign</span>
          <strong>{current.label}</strong>
          <p>{running ? `${liveConfidence}% confidence` : "Start camera to read live signs"}</p>
        </div>

        <div className="sentence">{sentence || "..."}</div>

        <div className="toolbar">
          <button title={running ? "Pause camera" : "Start camera"} onClick={running ? stopCamera : startCamera}>
            {running ? <Pause /> : <Play />}
          </button>
          <button title="Clear sentence" disabled={!sentence} onClick={() => setSentence("")}>
            <Trash2 />
          </button>
          <button title="Copy sentence" disabled={!sentence} onClick={() => navigator.clipboard.writeText(sentence)}>
            <Copy />
          </button>
          <button title="Speak current text or result" disabled={!canSpeak} onClick={speakCurrent}>
            <Volume2 />
          </button>
          <button title="Reset predictions" onClick={resetPredictions}>
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
            <p>{uploadResult ? `${uploadConfidence}% confidence${uploadAccepted ? "" : " · below threshold"}` : "No image selected"}</p>
          </div>
        </section>

        <label className="slider">
          <span>Acceptance threshold {Math.round(threshold * 100)}%</span>
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
          <span>Speak accepted results</span>
        </label>

        <div className="topList">
          {(uploadResult?.predictions || current.predictions || []).map((item) => (
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
