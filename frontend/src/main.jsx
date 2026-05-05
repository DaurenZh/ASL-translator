import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { Camera, Copy, ImageUp, Pause, Play, RotateCcw, Trash2, Volume2 } from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "";
const EMPTY_RESULT = { label: "-", confidence: 0, predictions: [] };
const SMOOTHING_WINDOW = 10;
const SMOOTHING_MIN_VOTES = 6;

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileRef = useRef(null);
  const streamRef = useRef(null);
  const lastAcceptedRef = useRef("");
  const recentPredictionsRef = useRef([]);
  const handLandmarkerRef = useRef(null);
  const mediaPipeReadyRef = useRef(false);
  const mediaPipeLoadingRef = useRef(false);
  const [running, setRunning] = useState(false);
  const [sentence, setSentence] = useState("");
  const [current, setCurrent] = useState(EMPTY_RESULT);
  const [threshold, setThreshold] = useState(0.92);
  const [voice, setVoice] = useState(false);
  const [status, setStatus] = useState("idle");
  const [uploadedImage, setUploadedImage] = useState("");
  const [uploadResult, setUploadResult] = useState(null);
  const [handCrop, setHandCrop] = useState("initializing");

  async function initMediaPipe() {
    if (mediaPipeReadyRef.current || mediaPipeLoadingRef.current) return;
    mediaPipeLoadingRef.current = true;
    setHandCrop("loading");

    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
      );
      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
      });
      mediaPipeReadyRef.current = true;
      setHandCrop("ready");
    } catch {
      setHandCrop("unavailable");
    } finally {
      mediaPipeLoadingRef.current = false;
    }
  }

  useEffect(() => {
    initMediaPipe();
  }, []);

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setRunning(true);
      setStatus("camera");
      initMediaPipe();
    } catch {
      setStatus("camera blocked");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setRunning(false);
    setStatus("paused");
    recentPredictionsRef.current = [];
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
    recentPredictionsRef.current = [];
    setCurrent(EMPTY_RESULT);
    setUploadResult(null);
    setUploadedImage("");
    setStatus(running ? "camera" : "idle");
    if (fileRef.current) fileRef.current.value = "";
  }

  function makeImageFromSource(source, landmarks = null) {
    const canvas = canvasRef.current;
    const width = source.videoWidth || source.naturalWidth || source.width || 640;
    const height = source.videoHeight || source.naturalHeight || source.height || 480;

    let sx = 0;
    let sy = 0;
    let sw = width;
    let sh = height;

    if (landmarks?.length) {
      const xs = landmarks.map((point) => point.x * width);
      const ys = landmarks.map((point) => point.y * height);
      const minX = Math.max(0, Math.min(...xs));
      const maxX = Math.min(width, Math.max(...xs));
      const minY = Math.max(0, Math.min(...ys));
      const maxY = Math.min(height, Math.max(...ys));
      const handSize = Math.max(maxX - minX, maxY - minY);
      const padding = handSize * 0.55;

      sx = Math.max(0, minX - padding);
      sy = Math.max(0, minY - padding);
      sw = Math.min(width - sx, maxX - minX + padding * 2);
      sh = Math.min(height - sy, maxY - minY + padding * 2);
    }

    canvas.width = 224;
    canvas.height = 224;
    canvas.getContext("2d").drawImage(source, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.86);
  }

  function getSmoothedLabel(result) {
    if (!result.ready || result.confidence < threshold || result.label === "nothing") {
      recentPredictionsRef.current = [];
      return "";
    }

    recentPredictionsRef.current = [
      ...recentPredictionsRef.current.slice(-(SMOOTHING_WINDOW - 1)),
      result.label,
    ];

    const counts = recentPredictionsRef.current.reduce((acc, label) => {
      acc[label] = (acc[label] || 0) + 1;
      return acc;
    }, {});
    const [label, count] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0] || [];
    return count >= SMOOTHING_MIN_VOTES ? label : "";
  }

  async function predictFrame() {
    if (!videoRef.current || !canvasRef.current || !running) return;
    const video = videoRef.current;
    if (!video.videoWidth || !video.videoHeight) return;

    let landmarks = null;
    if (mediaPipeReadyRef.current && handLandmarkerRef.current) {
      const detection = handLandmarkerRef.current.detectForVideo(video, performance.now());
      landmarks = detection.landmarks?.[0] || null;
      setHandCrop(landmarks ? "hand detected" : "no hand");
    }

    const image = makeImageFromSource(video, landmarks);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image }),
      });
      const result = await response.json();
      setCurrent(result);
      setStatus(result.ready ? (landmarks ? "live hand crop" : "live full frame") : "model missing");
      const acceptedLabel = getSmoothedLabel(result);
      if (acceptedLabel) applyToken(acceptedLabel);
      if (!acceptedLabel && result.confidence < threshold) lastAcceptedRef.current = "";
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
  const liveAccepted = Boolean(current.ready && current.confidence >= threshold);

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
          <p>
            {running
              ? `${liveConfidence}% confidence · ${liveAccepted ? "above threshold" : "below threshold"} · ${handCrop}`
              : "Start camera to read live signs"}
          </p>
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
