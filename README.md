# AirDraw Web

A browser-based version of the AirDraw desktop application, built with plain HTML, CSS and JavaScript. No frontend framework and no Node.js backend are required.

## Features

- Live webcam hand tracking with MediaPipe Hand Landmarker
- Pinch thumb + index finger to draw
- Index-finger gesture to erase
- Hand landmark and cursor overlay
- Brush color and size controls
- Clear canvas and export drawing as PNG
- Mouse-accessible controls
- Responsive layout

## Requirements

- A modern browser (Chrome or Edge recommended)
- Webcam
- Internet connection on first run / while using the CDN-hosted MediaPipe library and model
- Camera permission
- Python 3 (only used here as a simple static file server; it is not an application backend)

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

Click **Start camera** and allow camera access.

You can also use another static server. Camera access generally requires a secure context: `localhost` or HTTPS.

## Project files

- `index.html` — page structure
- `styles.css` — interface styling
- `app.js` — webcam, MediaPipe, gesture detection and drawing logic

## Gesture behaviour

- Pinch: draws continuously. Pinch state uses hysteresis to reduce flicker.
- Index finger up without pinching: gesture eraser.
- All fingers up: idle.
- The Brush / Eraser buttons and color swatches can also be used with the mouse.

## Notes

MediaPipe Tasks Vision is imported directly from jsDelivr as an ES module. The Hand Landmarker model is loaded from Google's public MediaPipe model storage. No webcam frames are uploaded by this app; hand detection is performed in the browser.

If the camera does not start:
1. Check browser camera permissions.
2. Close other apps that may be using the webcam.
3. Confirm the page is opened at `http://localhost:8000` or HTTPS, not directly as a `file://` page.
4. Confirm internet access to jsDelivr and Google model storage.

## Tuning

In `app.js`:
- `PINCH_START` controls how close thumb and index must be to begin drawing.
- `PINCH_RELEASE` controls when the pinch ends.
- `SMOOTHING` controls cursor smoothing. `0.60` mirrors the desktop version's original smoothing factor.
