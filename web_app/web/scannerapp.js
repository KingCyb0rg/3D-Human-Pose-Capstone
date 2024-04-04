function createOpenCVEnvironment() {
    const videoElement = document.createElement("video");
    document.append(video);
}

function testElementAppend() {
    const para = document.createElement("p");
    const text = document.createTextNode("This is a test, please work!");
    para.appendChild(text);
}