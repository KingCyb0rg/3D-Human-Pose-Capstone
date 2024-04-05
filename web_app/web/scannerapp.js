// Constants
const videoInput = document.getElementById("videoInput");
const canvasFrame = document.getElementById("canvasFrame");
const canvasOutput = document.getElementById("canvasOutput");
const FPS = 30;

// Initialize OpenCV.js
cv['onRuntimeInitialized'] = () => {
    startCapture();
};

// Function to start capturing video
function startCapture() {
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(function(stream) {
            videoInput.srcObject = stream;
            videoInput.play();
            processVideo();
        })
        .catch(function(err) {
            console.log("An error occurred! " + err);
        });
}

// Function to process video frames
function processVideo() {
    let src = new cv.Mat(videoInput.videoHeight, videoInput.videoWidth, cv.CV_8UC4);
    let dst = new cv.Mat(videoInput.videoHeight, videoInput.videoWidth, cv.CV_8UC1);
    let cap = new cv.VideoCapture(videoInput);

    let begin = Date.now();

    cap.read(src);
    cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
    cv.imshow("canvasOutput", dst);

    let delay = 1000 / FPS - (Date.now() - begin);
    setTimeout(processVideo, delay);
}