import { FilesetResolver, HandLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";

// ==========================================
// DOM ELEMENTS
// ==========================================

const video = document.getElementById("webcam");
const stage = document.getElementById("stage");
const paintCanvas = document.getElementById("paintCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");

const paintCtx = paintCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

const startButton = document.getElementById("startButton");
const statusPill = document.querySelector(".status-pill");
const statusText = document.getElementById("statusText");
const placeholder = document.getElementById("cameraPlaceholder");
const trackingLabel = document.getElementById("trackingLabel");
const handLabel = document.getElementById("handLabel");
const fpsLabel = document.getElementById("fpsLabel");
const modeChip = document.getElementById("modeChip");
const brushSizeInput = document.getElementById("brushSize");
const brushValue = document.getElementById("brushValue");
const swatches = [...document.querySelectorAll(".swatch")];

// ==========================================
// APPLICATION STATE
// ==========================================

let landmarker = null;
let cameraStream = null;
let animationId = null;

let lastVideoTime = -1;
let lastFrameAt = performance.now();
let frameCounter = 0;

let brushSize = Number(brushSizeInput.value);
let brushColor = "#00e676";
let selectedTool = "brush";

let pinchActive = false;
let previousPoint = null;
let cursor = null;
let previousMode = null;

const PINCH_START = 0.060;
const PINCH_RELEASE = 0.078;
const SMOOTHING = 0.60;

// ==========================================
// MOBILE CAMERA FIX
// ==========================================

video.style.objectFit = "contain";
video.style.objectPosition = "center";
video.style.transform = "scaleX(-1)";
video.style.width = "100%";
video.style.height = "100%";

video.setAttribute("playsinline", "");
video.setAttribute("webkit-playsinline", "");

video.muted = true;
video.autoplay = true;

stage.style.aspectRatio = "16 / 9";

// ==========================================
// STATUS
// ==========================================

function setStatus(message, ready = false) {
  statusText.textContent = message;
  statusPill.classList.toggle("ready", ready);
}

// ==========================================
// CANVAS RESIZING
// ==========================================

function resizeCanvases() {
  const width = stage.clientWidth;
  const height = stage.clientHeight;
  const dpr = window.devicePixelRatio || 1;

  if (!width || !height) return;

  const targetWidth = Math.round(width * dpr);
  const targetHeight = Math.round(height * dpr);

  if (
    paintCanvas.width === targetWidth &&
    paintCanvas.height === targetHeight &&
    paintCanvas.style.width === `${width}px` &&
    paintCanvas.style.height === `${height}px`
  ) {
    return;
  }

  // Preserve existing drawing.
  const oldCanvas = document.createElement("canvas");

  oldCanvas.width = paintCanvas.width;
  oldCanvas.height = paintCanvas.height;

  if (oldCanvas.width && oldCanvas.height) {
    oldCanvas.getContext("2d").drawImage(
      paintCanvas,
      0,
      0
    );
  }

  for (const canvas of [paintCanvas, overlayCanvas]) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;

    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }

  const scaleX = targetWidth / width;
  const scaleY = targetHeight / height;

  paintCtx.setTransform(scaleX, 0, 0, scaleY, 0, 0);
  overlayCtx.setTransform(scaleX, 0, 0, scaleY, 0, 0);

  if (oldCanvas.width && oldCanvas.height) {
    paintCtx.drawImage(
      oldCanvas,
      0,
      0,
      width,
      height
    );
  }

  previousPoint = null;
  cursor = null;
}

// ==========================================
// CAMERA ASPECT RATIO
// ==========================================

function syncCameraLayout() {
  if (video.videoWidth && video.videoHeight) {
    stage.style.aspectRatio =
      `${video.videoWidth} / ${video.videoHeight}`;
  }

  resizeCanvases();
}

// ==========================================
// HAND COORDINATE MAPPING
// ==========================================

function getStagePoint(landmark) {
  const width = stage.clientWidth;
  const height = stage.clientHeight;

  const videoWidth = video.videoWidth || width;
  const videoHeight = video.videoHeight || height;

  if (!videoWidth || !videoHeight) {
    return { x: 0, y: 0 };
  }

  const scale = Math.min(
    width / videoWidth,
    height / videoHeight
  );

  const displayedWidth = videoWidth * scale;
  const displayedHeight = videoHeight * scale;

  const offsetX = (width - displayedWidth) / 2;
  const offsetY = (height - displayedHeight) / 2;

  return {
    x: offsetX + (1 - landmark.x) * displayedWidth,
    y: offsetY + landmark.y * displayedHeight
  };
}

// ==========================================
// GESTURE DETECTION
// ==========================================

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function getFingerStates(landmarks, handedness) {
  const isRight = handedness === "Right";

  return {
    thumb: isRight
      ? landmarks[4].x < landmarks[3].x
      : landmarks[4].x > landmarks[3].x,

    index: landmarks[8].y < landmarks[6].y,

    middle: landmarks[12].y < landmarks[10].y,

    ring: landmarks[16].y < landmarks[14].y,

    pinky: landmarks[20].y < landmarks[18].y
  };
}

// ==========================================
// DRAWING TOOLS
// ==========================================

function setTool(tool) {
  selectedTool = tool;

  document.getElementById("brushTool")
    .classList.toggle("active", tool === "brush");

  document.getElementById("eraserTool")
    .classList.toggle("active", tool === "eraser");

  previousPoint = null;
  previousMode = null;
}

function chooseColor(color) {
  brushColor = color;

  setTool("brush");

  swatches.forEach(button => {
    button.classList.toggle(
      "selected",
      button.dataset.color === color
    );
  });
}

function clearCanvas() {
  paintCtx.clearRect(
    0,
    0,
    stage.clientWidth,
    stage.clientHeight
  );

  previousPoint = null;
}

// ==========================================
// SAVE DRAWING
// ==========================================

function saveCanvas() {
  const width = stage.clientWidth;
  const height = stage.clientHeight;

  const exportCanvas = document.createElement("canvas");

  exportCanvas.width = width * 2;
  exportCanvas.height = height * 2;

  const ctx = exportCanvas.getContext("2d");

  ctx.fillStyle = "#ffffff";

  ctx.fillRect(
    0,
    0,
    exportCanvas.width,
    exportCanvas.height
  );

  ctx.drawImage(
    paintCanvas,
    0,
    0,
    exportCanvas.width,
    exportCanvas.height
  );

  const link = document.createElement("a");

  link.download = `airdraw-${Date.now()}.png`;
  link.href = exportCanvas.toDataURL("image/png");

  link.click();
}

// ==========================================
// DRAW CURSOR
// ==========================================

function drawCursor(point, pinching) {
  overlayCtx.beginPath();

  overlayCtx.arc(
    point.x,
    point.y,
    pinching ? 10 : 8,
    0,
    Math.PI * 2
  );

  overlayCtx.lineWidth = 2;

  overlayCtx.strokeStyle = pinching
    ? "#00e676"
    : "#ffad55";

  overlayCtx.fillStyle = pinching
    ? "rgba(0,230,118,.15)"
    : "rgba(255,173,85,.12)";

  overlayCtx.fill();
  overlayCtx.stroke();
}

// ==========================================
// DRAW HAND SKELETON
// ==========================================

function drawHandSkeleton(landmarks) {
  const connections = HandLandmarker.HAND_CONNECTIONS;

  overlayCtx.lineWidth = 2;
  overlayCtx.strokeStyle = "rgba(102,190,255,.78)";

  for (const connection of connections) {
    const a = getStagePoint(
      landmarks[connection.start]
    );

    const b = getStagePoint(
      landmarks[connection.end]
    );

    overlayCtx.beginPath();
    overlayCtx.moveTo(a.x, a.y);
    overlayCtx.lineTo(b.x, b.y);
    overlayCtx.stroke();
  }

  for (const landmark of landmarks) {
    const point = getStagePoint(landmark);

    overlayCtx.beginPath();

    overlayCtx.arc(
      point.x,
      point.y,
      2.4,
      0,
      Math.PI * 2
    );

    overlayCtx.fillStyle = "#e8f4ff";
    overlayCtx.fill();
  }
}

// ==========================================
// DRAW STROKE
// ==========================================

function drawStroke(point, erase = false) {
  const mode = erase ? "erase" : "draw";

  if (previousMode !== mode) {
    previousPoint = null;
  }

  previousMode = mode;

  paintCtx.globalCompositeOperation = erase
    ? "destination-out"
    : "source-over";

  paintCtx.strokeStyle = erase
    ? "#000000"
    : brushColor;

  paintCtx.fillStyle = erase
    ? "#000000"
    : brushColor;

  paintCtx.lineWidth = erase
    ? brushSize * 2
    : brushSize;

  paintCtx.lineCap = "round";
  paintCtx.lineJoin = "round";

  if (!previousPoint) {
    paintCtx.beginPath();

    paintCtx.arc(
      point.x,
      point.y,
      paintCtx.lineWidth / 2,
      0,
      Math.PI * 2
    );

    paintCtx.fill();
  } else {
    paintCtx.beginPath();

    paintCtx.moveTo(
      previousPoint.x,
      previousPoint.y
    );

    paintCtx.lineTo(point.x, point.y);
    paintCtx.stroke();
  }

  paintCtx.globalCompositeOperation = "source-over";

  previousPoint = point;
}

// ==========================================
// HAND TRACKING LOOP
// ==========================================

function handleFrame() {
  if (!landmarker || video.readyState < 2) {
    animationId = requestAnimationFrame(handleFrame);
    return;
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const result = landmarker.detectForVideo(
      video,
      performance.now()
    );

    overlayCtx.clearRect(
      0,
      0,
      stage.clientWidth,
      stage.clientHeight
    );

    const hasHand =
      result.landmarks &&
      result.landmarks.length > 0;

    if (hasHand) {
      const landmarks = result.landmarks[0];

      const handedness =
        result.handedness?.[0]?.[0]?.categoryName ||
        "Unknown";

      const pinchDistance = distance(
        landmarks[4],
        landmarks[8]
      );

      if (!pinchActive && pinchDistance < PINCH_START) {
        pinchActive = true;
      } else if (
        pinchActive &&
        pinchDistance > PINCH_RELEASE
      ) {
        pinchActive = false;
      }

      const finger = getFingerStates(
        landmarks,
        handedness
      );

      const allUp = Object.values(finger).every(Boolean);

      const point = getStagePoint(landmarks[8]);

      if (!cursor) {
        cursor = point;
      } else {
        cursor = {
          x: SMOOTHING * cursor.x +
            (1 - SMOOTHING) * point.x,

          y: SMOOTHING * cursor.y +
            (1 - SMOOTHING) * point.y
        };
      }

      drawHandSkeleton(landmarks);
      drawCursor(cursor, pinchActive);

      const isGestureErasing =
        finger.index &&
        !pinchActive &&
        !allUp;

      if (pinchActive) {
        const erase = selectedTool === "eraser";

        drawStroke(cursor, erase);

        modeChip.textContent = erase
          ? "ERASING"
          : "DRAWING";

        modeChip.style.color = erase
          ? "#ffb86b"
          : "#00e676";

      } else if (
        selectedTool === "eraser" ||
        isGestureErasing
      ) {
        if (cursor.y > 64) {
          drawStroke(cursor, true);
        } else {
          previousPoint = null;
        }

        modeChip.textContent = "ERASING";
        modeChip.style.color = "#ffb86b";

      } else {
        previousPoint = null;
        previousMode = null;

        modeChip.textContent = allUp
          ? "IDLE"
          : "TRACKING";

        modeChip.style.color = "#91a0b7";
      }

      trackingLabel.textContent = pinchActive
        ? "HAND: PINCH DETECTED"
        : "HAND: TRACKING";

      handLabel.textContent = `${handedness} hand`;

    } else {
      pinchActive = false;
      previousPoint = null;
      previousMode = null;
      cursor = null;

      trackingLabel.textContent = "HAND: NOT DETECTED";
      handLabel.textContent = "No hand";

      modeChip.textContent = "IDLE";
      modeChip.style.color = "#91a0b7";
    }

    // FPS calculation.
    frameCounter++;

    const now = performance.now();
    const elapsed = now - lastFrameAt;

    if (elapsed >= 1000) {
      const fps = Math.round(
        frameCounter * 1000 / elapsed
      );

      fpsLabel.textContent = `${fps} FPS`;

      frameCounter = 0;
      lastFrameAt = now;
    }
  }

  animationId = requestAnimationFrame(handleFrame);
}

// ==========================================
// INITIALIZE MEDIAPIPE
// ==========================================

async function initializeModel() {
  if (landmarker) return;

  const vision = await FilesetResolver.forVisionTasks(
    WASM_URL
  );

  const options = {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU"
    },

    runningMode: "VIDEO",
    numHands: 1,

    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  };

  try {
    landmarker = await HandLandmarker.createFromOptions(
      vision,
      options
    );
  } catch (error) {
    // CPU fallback for unsupported mobile GPUs.
    console.warn("GPU unavailable. Using CPU.", error);

    options.baseOptions.delegate = "CPU";

    landmarker = await HandLandmarker.createFromOptions(
      vision,
      options
    );
  }
}

// ==========================================
// START CAMERA
// ==========================================

async function startCamera() {
  startButton.disabled = true;
  startButton.textContent = "Loading model...";

  setStatus("Loading hand tracking model");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "Camera access requires HTTPS or localhost."
      );
    }

    await initializeModel();

    startButton.textContent = "Starting camera...";

    setStatus("Starting camera");

    cameraStream =
      await navigator.mediaDevices.getUserMedia({
        audio: false,

        video: {
          facingMode: {
            ideal: "user"
          },

          width: {
            ideal: 960
          },

          height: {
            ideal: 540
          },

          frameRate: {
            ideal: 30,
            max: 30
          }
        }
      });

    video.srcObject = cameraStream;

    await video.play();

    syncCameraLayout();

    console.log(
      "Camera resolution:",
      video.videoWidth,
      "x",
      video.videoHeight
    );

    placeholder.classList.add("hidden");

    setStatus("Camera active", true);

    startButton.textContent = "Camera running";
    startButton.disabled = true;

    lastVideoTime = -1;
    lastFrameAt = performance.now();
    frameCounter = 0;

    if (animationId) {
      cancelAnimationFrame(animationId);
    }

    handleFrame();

  } catch (error) {
    console.error(error);

    if (cameraStream) {
      cameraStream.getTracks().forEach(track => {
        track.stop();
      });

      cameraStream = null;
      video.srcObject = null;
    }

    setStatus("Unable to start");

    startButton.disabled = false;
    startButton.textContent = "Try again";

    alert(
      "AirDraw could not start:\n" +
      error.message +
      "\n\nPlease allow camera access and use HTTPS."
    );
  }
}

// ==========================================
// BUTTON EVENT LISTENERS
// ==========================================

startButton.addEventListener("click", startCamera);

brushSizeInput.addEventListener("input", () => {
  brushSize = Number(brushSizeInput.value);

  brushValue.textContent = `${brushSize} px`;
});

swatches.forEach(button => {
  button.addEventListener("click", () => {
    chooseColor(button.dataset.color);
  });
});

document.getElementById("brushTool")
  .addEventListener("click", () => {
    setTool("brush");
  });

document.getElementById("eraserTool")
  .addEventListener("click", () => {
    setTool("eraser");
  });

document.getElementById("clearButton")
  .addEventListener("click", clearCanvas);

document.getElementById("saveButton")
  .addEventListener("click", saveCanvas);

// ==========================================
// RESPONSIVE EVENT LISTENERS
// ==========================================

video.addEventListener(
  "loadedmetadata",
  syncCameraLayout
);

video.addEventListener(
  "resize",
  syncCameraLayout
);

window.addEventListener(
  "resize",
  syncCameraLayout
);

window.addEventListener("orientationchange", () => {
  setTimeout(syncCameraLayout, 300);
});

const stageObserver = new ResizeObserver(() => {
  resizeCanvases();
});

stageObserver.observe(stage);

// ==========================================
// CLEANUP
// ==========================================

window.addEventListener("beforeunload", () => {
  if (animationId) {
    cancelAnimationFrame(animationId);
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach(track => {
      track.stop();
    });
  }

  if (landmarker) {
    landmarker.close();
  }

  stageObserver.disconnect();
});

// ==========================================
// INITIALIZATION
// ==========================================

resizeCanvases();

console.log("AirDraw initialized successfully");

// END OF APP.JS