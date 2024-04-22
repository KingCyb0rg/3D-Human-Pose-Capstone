const recordingTimeMS = 3000;
let recordingOutput = undefined;
const video = document.getElementById("webcam");

function wait(delay) {
    return new Promise((resolve) => setTimeout(resolve, delay));
}

function startRecording(stream, length) {

    let recorder = new MediaRecorder(stream);
    let data = [];

    recorder.ondatavailable = (event) => data.push(event.data);
    recorder.start();

    let stopped = new Promise((resolve, reject) => {
        recorder.onstop = resolve;
        recorder.onerror = (event) => reject(event.name);
    });

    let recorded = wait(length).then(() => {
        if (recorder.state == "recording") {
            recorder.stop();
        }
    });

    return Promise.all([stopped, recorded]).then(() => data);
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
            video.captureStream =
                video.captureStream || video.mozCaptureStream;
            return new Promise((resolve) => (video.onplaying = resolve));
        })
        .then(() => startRecording(video.captureStream(), recordingTimeMS))
        .then((recordedChunks) => {
            console.log("Capturing recording chunks");
            let recordedBlob = new Blob(recordedChunks, { type: "video/mov" });
            recordedBlob.lastModifiedDate = new Date();
            recordedBlob.name = "output";
            recordingOutput = URL.createObjectURL(recordedBlob);
            console.log("Created video output");
        })
        .catch((error) => {
            if (error.name == "NotFoundError") {
                console.log("Camera not found. Can't record.");
            }
            else {
                console.log(error);
            }
        });

        return recordingOutput;
}

startRecordingOnClick();