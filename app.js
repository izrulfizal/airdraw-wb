
import { FilesetResolver, HandLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";

// ======================================================
// DOM ELEMENTS
// ======================================================

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

// ======================================================
// MOBILE CAMERA FIX
// ======================================================

// Override existing CSS to prevent camera distortion.
video.style.objectFit = "contain";
video.style.objectPosition = "center";
video.style.transform = "scaleX(-1)";
video.style.width = "100%";
video.style.height = "100%";

video.setAttribute("playsinline", "");
video.setAttribute("webkit-playsinline", "");
video.muted = true;
video.autoplay = true;

// Default aspect ratio before the camera starts.
stage.style.aspectRatio = "var(--camera-ratio, 16 / 9)";

// ======================================================
// APPLICATION STATE
// ======================================================

let landmarker = null;
let cameraStream = null;
let animationId = null;

let lastVideoTime = -1;
let lastFrameAt = performance.now();
let frameCounter = 0;
let fps = 0;

let brushSize = Number(brushSizeInput.value);
let brushColor = "#00e676";
let selectedTool = "brush";

let pinchActive = false;
let previousPoint = null;
let cursor = null;
let currentHandedness = "Unknown";

const PINCH_START = 0.060;
const PINCH_RELEASE = 0.078;
const SMOOTHING = 0.60;

// ======================================================
// STATUS
// ======================================================

function setStatus(message, ready = false) {
  statusText.textContent = message;
  statusPill.classList.toggle("ready", ready);
}

// ======================================================
// CANVAS DIMENSIONS
// ======================================================

function getStageSize() {
  return {
    width: stage.clientWidth,
    height: stage.clientHeight
  };
}

function resizeCanvases() {
  const { width, height } = getStageSize();

  if (!width || !height) return;

  const dpr = window.devicePixelRatio || 1;

  for (const canvas of [paintCanvas, overlayCanvas]) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);

    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }

  paintCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  previousPoint = null;
  cursor = null;
}

// ======================================================
// RESPONSIVE CAMERA LAYOUT
// ======================================================

function syncCameraLayout() {
  // Use the actual camera resolution instead of
  // forcing a fixed landscape aspect ratio.

  if (video.videoWidth && video.videoHeight) {
    stage.style.setProperty(
      "--camera-ratio",
      `${video.videoWidth} / ${video.videoHeight}`
    );
  }

  const { width, height } = getStageSize();
  const dpr = window.devicePixelRatio || 1;

  if (!width || !height) return;

  // Avoid unnecessary canvas resizing.
  if (
    paintCanvas.width === Math.round(width * dpr) &&
    paintCanvas.height === Math.round(height * dpr) &&
    paintCanvas.style.width === `${width}px` &&
    paintCanvas.style.height === `${height}px`
  ) {
    return;
  }

  // Preserve existing drawings during resizing.
  const oldCanvas = document.createElement("canvas");

  oldCanvas.width = paintCanvas.width;
  oldCanvas.height = paintCanvas.height;

  if (oldCanvas.width && oldCanvas.height) {
    oldCanvas
      .getContext("2d")
      .drawImage(paintCanvas, 0, 0);
  }

  resizeCanvases();

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

// ======================================================
// HAND LANDMARK COORDINATE MAPPING
// ======================================================

function getStagePoint(landmark) {
  const { width, height } = getStageSize();

  const videoWidth = video.videoWidth || width;
  const videoHeight = video.videoHeight || height;

  if (!videoWidth || !videoHeight) {
    return { x: 0, y: 0 };
  }

  // Match the behaviour of object-fit: contain.
  const scale = Math.min(
    width / videoWidth,
    height / videoHeight
  );

  const displayedWidth = videoWidth * scale;
  const displayedHeight = videoHeight * scale;

  // Account for letterboxing.
  const offsetX = (width - displayedWidth) / 2;
  const offsetY = (height - displayedHeight) / 2;

  // Mirror X because the camera preview is mirrored.
  const x =
    offsetX + (1 - landmark.x) * displayedWidth;

  const y =
    offsetY + landmark.y * displayedHeight;

  return { x, y };
}

// ======================================================
// GESTURE DETECTION
// ======================================================

function distance(a, b) {
  return Math.hypot(
    a.x - b.x,
    a.y - b.y
  );
}

function getFingerStates(landmarks, handedness) {
  const tip = {
    thumb: 4,
    index: 8,
    middle: 12,
    ring: 16,
    pinky: 20
  };

  const pip = {
    thumb: 3,
    index: 6,
    middle: 10,
    ring: 14,
    pinky: 18
  };

  const isRight = handedness === "Right";

  const thumbUp = isRight
    ? landmarks[tip.thumb].x < landmarks[pip.thumb].x
    : landmarks[tip.thumb].x > landmarks[pip.thumb].x;

  return {
    thumb: thumbUp,

    index:
      landmarks[tip.index].y <
      landmarks[pip.index].y,

    middle:
      landmarks[tip.middle].y <
      landmarks[pip.middle].y,

    ring:
      landmarks[tip.ring].y <
      landmarks[pip.ring].y,

    pinky:
      landmarks[tip.pinky].y <
      landmarks[pip.pinky].y
  };
}

// ======================================================
// DRAWING TOOLS
// ======================================================

function setTool(tool) {
  selectedTool = tool;

  document
    .getElementById("brushTool")
    .classList.toggle("active", tool === "brush");

  document
    .getElementById("eraserTool")
    .classList.toggle("active", tool === "eraser");

  previousPoint = null;
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
  const { width, height } = getStageSize();

  paintCtx.clearRect(0, 0, width, height);

  previousPoint = null;
}

function saveCanvas() {
  const { width, height } = getStageSize();

  const exportCanvas = document.createElement("canvas");
  const scale = 2;

  exportCanvas.width = Math.round(width * scale);
  exportCanvas.height = Math.round(height * scale);

  const ctx = exportCanvas.getContext("2d");

  ctx.fillStyle = "#ffffff";

  ctx.fillRect(
    0,
    0,
    exportCanvas.width,
    exportCanvas.height
  );

  ctx.scale(scale, scale);

  ctx.drawImage(
    paintCanvas,
    0,
    0,
    width,
    height
  );

  const link = document.createElement("a");

  link.download =
    `airdraw-${new Date()
      .toISOString()
      .replace(/[:.]/g, "-")}.png`;

  link.href = exportCanvas.toDataURL("image/png");

  link.click();
}

// ======================================================
// DRAW CURSOR
// ======================================================

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

// ======================================================
// DRAW HAND SKELETON
// ======================================================

function drawHandSkeleton(landmarks) {
  const connections = HandLandmarker.HAND_CONNECTIONS;

  overlayCtx.lineWidth = 2;

  overlayCtx.strokeStyle =
    "rgba(102, 190, 255, .78)";

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

// ======================================================
// DRAW STROKE
// ======================================================

function drawStroke(
  point,
  erase = selectedTool === "eraser"
) {
  if (!previousPoint) {
    previousPoint = point;

    paintCtx.beginPath();

    paintCtx.arc(
      point.x,
      point.y,
      brushSize / 2,
      0,
      Math.PI * 2
    );

    paintCtx.fillStyle = erase
      ? "#000000"
      : brushColor;

    paintCtx.globalCompositeOperation = erase
      ? "destination-out"
      : "source-over";

    paintCtx.fill();

    paintCtx.globalCompositeOperation =
      "source-over";

    return;
  }

  paintCtx.beginPath();

  paintCtx.moveTo(
    previousPoint.x,
    previousPoint.y
  );

  paintCtx.lineTo(point.x, point.y);

  paintCtx.lineCap = "round";
  paintCtx.lineJoin = "round";

  paintCtx.lineWidth = erase
    ? brushSize * 2
    : brushSize;

  if (erase) {
    paintCtx.globalCompositeOperation =
      "destination-out";

    paintCtx.strokeStyle = "rgba(0,0,0,1)";
  } else {
    paintCtx.globalCompositeOperation =
      "source-over";

    paintCtx.strokeStyle = brushColor;
  }

  paintCtx.stroke();

  paintCtx.globalCompositeOperation =
    "source-over";

  previousPoint = point;
}

// ======================================================
// MAIN HAND TRACKING LOOP
// ======================================================

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

    const { width, height } = getStageSize();

    overlayCtx.clearRect(0, 0, width, height);

    const hasHand =
      result.landmarks &&
      result.landmarks.length > 0;

    if (hasHand) {
      const landmarks = result.landmarks[0];

      currentHandedness =
        result.handedness?.[0]?.[0]?.categoryName ||
        "Unknown";

      // Detect pinch gesture.
      const pinchDistance = distance(
        landmarks[4],
        landmarks[8]
      );

      if (
        !pinchActive &&
        pinchDistance < PINCH_START
      ) {
        pinchActive = true;
      } else if (
        pinchActive &&
        pinchDistance > PINCH_RELEASE
      ) {
        pinchActive = false;
      }

      const finger = getFingerStates(
        landmarks,
        currentHandedness
      );

      const allUp = Object.values(finger).every(Boolean);

      const point = getStagePoint(landmarks[8]);

      // Smooth cursor movement.
      if (!cursor) {
        cursor = point;
      } else {
        cursor = {
          x:
            SMOOTHING * cursor.x +
            (1 - SMOOTHING) * point.x,

          y:
            SMOOTHING * cursor.y +
            (1 - SMOOTHING) * point.y
        };
      }

      drawHandSkeleton(landmarks);
      drawCursor(cursor, pinchActive);

      const isDrawing = pinchActive;

      const isGestureErasing =
        finger.index &&
        !pinchActive &&
        !allUp;

      if (isDrawing) {
        drawStroke(cursor);

        modeChip.textContent = "DRAWING";
        modeChip.style.color = "#00e676";

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

        modeChip.textContent = allUp
          ? "IDLE"
          : "TRACKING";

        modeChip.style.color = "#91a0b7";
      }

      trackingLabel.textContent = pinchActive
        ? "HAND: PINCH DETECTED"
        : "HAND: TRACKING";

      handLabel.textContent =
        `${currentHandedness} hand`;

    } else {
      pinchActive = false;
      previousPoint = null;
      cursor = null;

      trackingLabel.textContent =
        "HAND: NOT DETECTED";

      handLabel.textContent = "No hand";

      modeChip.textContent = "IDLE";
      modeChip.style.color = "#91a0b7";
    }

    // Calculate FPS.
    frameCounter++;

    const now = performance.now();

    if (now - lastFrameAt >= 1000) {
      fps = Math.round(
        frameCounter * 1000 /
        (now - lastFrameAt)
      );

      fpsLabel.textContent = `${fps} FPS`;

      frameCounter = 0;
      lastFrameAt = now;
    }
  }

  animationId = requestAnimationFrame(handleFrame);
}

// ======================================================
// START CAMERA
// ======================================================

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

    // Initialize MediaPipe.
    if (!landmarker) {
      const vision =
        await FilesetResolver.forVisionTasks(WASM_URL);

      landmarker =
        await HandLandmarker.createFromOptions(
          vision,
          {
            baseOptions: {
              modelAssetPath: MODEL_URL,
              delegate: "GPU"
            },

            runningMode: "VIDEO",
            numHands: 1,

            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
          }
        );
    }

    // Request the front-facing camera.
    cameraStream =
      await navigator.mediaDevices.getUserMedia({
        audio: false,

        video: {
          facingMode: "user",

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

    // Adapt the layout to the actual camera dimensions.
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

    handleFrame();

  } catch (error) {
    console.error(error);

    if (cameraStream) {
      cameraStream
        .getTracks()
        .forEach(track => track.stop());

      cameraStream = null;
      video.srcObject = null;
    }

    setStatus("Unable to start");

    startButton.disabled = false;
    startButton.textContent = "Try again";

    alert(
      `AirDraw could not start:\n${error.message}` +
      "\n\nAllow camera access and run this " +
      "project from localhost or HTTPS."
    );
  }
}

// ======================================================
// EVENT LISTENERS
// ======================================================

startButton.addEventListener(
  "click",
  startCamera
);

brushSizeInput.addEventListener("input", () => {
  brushSize = Number(brushSizeInput.value);
  brushValue.textContent = `${brushSize} px`;
});

swatches.forEach(button => {
  button.addEventListener("click", () => {
    chooseColor(button.dataset.color);
  });
});

document
  .getElementById("brushTool")
  .addEventListener("click", () => {
    setTool("brush");
  });

document
  .getElementById("eraserTool")
  .addEventListener("click", () => {
    setTool("eraser");
  });

document
  .getElementById("clearButton")
  .addEventListener("click", clearCanvas);

document
  .getElementById("saveButton")
  .addEventListener("click", saveCanvas);

// ======================================================
// MOBILE RESPONSIVENESS AND SCREEN ROTATION
// ======================================================

// Camera metadata becomes available.
video.addEventListener(
  "loadedmetadata",
  syncCameraLayout
);

// Camera resolution changes.
video.addEventListener(
  "resize",
  syncCameraLayout
);

// Browser window changes.
window.addEventListener(
  "resize",
  syncCameraLayout
);

// Mobile device orientation changes.
window.addEventListener("orientationchange", () => {
  setTimeout(syncCameraLayout, 300);
});

// Detect changes to the stage container.
const stageObserver = new ResizeObserver(() => {
  syncCameraLayout();
});

stageObserver.observe(stage);

// ======================================================
// CLEANUP
// ======================================================

window.addEventListener("beforeunload", () => {
  if (animationId) {
    cancelAnimationFrame(animationId);
  }

  if (cameraStream) {
    cameraStream
      .getTracks()
      .forEach(track => track.stop());
  }

  if (landmarker) {
    landmarker.close();
  }

  stageObserver.disconnect();
});

// ======================================================
// INITIALIZATION
// ======================================================

resizeCanvases();
`