import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";


const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const callibrationButton = document.getElementById("callibrationBtn");
const recordingButton = document.getElementById("recordBtn");
const uploadButton = document.getElementById("uploadBtn");
const stopButton = document.getElementById("stopBtn");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
const constraints = { video: true };
const recordingTimeMS = 30000;


let poseLandmarker = undefined;
let runningMode = "VIDEO";
let webcamRunning = true;
document.body.style.backgroundColor = "red";
let recordingOutput = undefined;
let flag = false;
let req;
var timeout;

function angleMeasure(shx, shy, wrx, wry, hpx, hpy) {
  var wrist = Math.atan2(wry - shy, wrx - shx);
  var hip = Math.atan2(hpy - shy, hpx - shx);
  return wrist - hip;
}

function isGood(leftAngle, rightAngle) {
  if (leftAngle < 1.5 || leftAngle > 2 || rightAngle < -1.8 || rightAngle > -1.5) {
    console.log("NO GOOD");
    clearTimeout(timeout);
    timeout = setTimeout(() => {flag = true;}, 5000);;
    document.body.style.backgroundColor = "red";;
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
      flag = isGood(left, right);
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
  if (webcamRunning === true && flag === false) {
    req = window.requestAnimationFrame(predictWebcam);
  }
  else {
    window.cancelAnimationFrame(req);
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.restore();
  }
}

function start_prediction() {
  createPoseLandmarker();
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

function wait(delay) {
  return new Promise((resolve) => setTimeout(resolve, delay));
}

function startRecording(stream, length) {

  let recorder = new MediaRecorder(stream);
  let data = [];

  recorder.ondataavailable = (event) => data.push(event.data);
  recorder.start();

  let stopped = new Promise((resolve, reject) => {
    recorder.onstop = resolve;
    recorder.onerror = (event) => reject(event.name);
  });

  let recorded = wait(length).then(() => {
    if (recorder.state === "recording") {
      recorder.stop();
    }
  });

  return Promise.all([stopped, recorded]).then(() => data);
}

function stop(stream) {
  stream.getTracks().forEach((track) => track.stop());
}

function startRecordingOnClick() {
  console.log("getting started");
  navigator.mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {
      console.log("getting video stream");
      video.srcObject = stream;
      uploadButton.href = stream;
      video.captureStream =
        video.captureStream || video.mozCaptureStream;
      return new Promise((resolve) => (video.onplaying = resolve));
    })
    .then(() => startRecording(video.captureStream(), recordingTimeMS))
    .then((recordedChunks) => {
      console.log("Captured recording chunks");
      let recordedBlob = new Blob(recordedChunks, { type: "video/mov" });
      recordingOutput = URL.createObjectURL(recordedBlob);
      uploadButton.href = recordingOutput;
      uploadButton.download = "RecordedVideo.webm";
      console.log("Created video output");
    })
    .catch((error) => {
      if (error.name === "NotFoundError") {
        console.log("Camera not found. Can't record.");
      }
      else {
        console.log(error);
      }
    });

  return recordingOutput;
}

function stopRecordingOnClick() {
  stop(video.srcObject);
  canvasCtx.clearRect(0, 0, canvasCtx.width, canvasCtx.width)
}

function upload() { }


stopButton.addEventListener("click", stopRecordingOnClick, false);
uploadButton.addEventListener("click", upload, false);
callibrationButton.addEventListener("click", start_prediction, false);
recordingButton.addEventListener("click", startRecordingOnClick, false);