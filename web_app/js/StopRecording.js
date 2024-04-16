const video = document.getElementById("webcam");


function stop(stream) {
    stream.getTrack().forEach((track) => track.stop());
  }

  function stopRecordingOnClick() {
    stop(video.srcObject);
  }

stopRecordingOnClick();