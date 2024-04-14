import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";


let poseLandmarker = undefined;
let runningMode = "VIDEO";
let webcamRunning = true;
const videoHeight = "360px";
const videoWidth = "480px";
document.body.style.backgroundColor = "red";
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
const constraints = {video: true};



function angleMeasure(shx, shy, wrx, wry, hpx, hpy) {
  var wrist = Math.atan2(wry - shy, wrx - shx);
  var hip = Math.atan2(hpy - shy, hpx - shx);
  return wrist - hip;
}

function isGood(leftAngle, rightAngle) {
  if (leftAngle < 1.5 || leftAngle > 2 || rightAngle < -1.8 || rightAngle > -1.5) {
    console.log("NO GOOD");
    document.body.style.backgroundColor = "red";
  }
  else {
    console.log("GOOD!!!!!!!!!!");
    document.body.style.backgroundColor = "green";
  }
}

const createPoseLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "app/shared/models/pose_landmarker_lite.task",
    },
    runningMode: runningMode,
  });
};

async function predictWebcam() {
  let lastVideoTime = -1;
  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;

  // Now let's start detecting the stream.
  let startTimeMs = performance.now();

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const poseLandMarkerResult = poseLandmarker.detectForVideo(
      video,
      startTimeMs);
    try {
      let left = angleMeasure(
        poseLandMarkerResult.landmarks[0][12].x,
        poseLandMarkerResult.landmarks[0][12].y,
        poseLandMarkerResult.landmarks[0][16].x,
        poseLandMarkerResult.landmarks[0][16].y,
        poseLandMarkerResult.landmarks[0][24].x,
        poseLandMarkerResult.landmarks[0][24].y
      )
      let right = angleMeasure(
        poseLandMarkerResult.landmarks[0][11].x,
        poseLandMarkerResult.landmarks[0][11].y,
        poseLandMarkerResult.landmarks[0][15].x,
        poseLandMarkerResult.landmarks[0][15].y,
        poseLandMarkerResult.landmarks[0][23].x,
        poseLandMarkerResult.landmarks[0][23].y
      )
      console.log("Left: " + left);
      console.log("Right: " + right);
      isGood(left, right);
    } catch (err) {
      canvasCtx.restore();
      console.log("OFF SCREEN!");
    }

    for (const landmarks of poseLandMarkerResult.landmarks) {
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      drawingUtils.drawConnectors(
        landmarks,
        PoseLandmarker.POSE_CONNECTIONS
      );
      drawingUtils.drawLandmarks(landmarks);
      canvasCtx.restore();
    }

  }

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}

function start_prediction() {
  createPoseLandmarker();
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

start_prediction();