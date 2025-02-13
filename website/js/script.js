// Function to start webcam stream for both sections
function startWebcam(videoId) {
    const video = document.getElementById(videoId);
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(error => {
            console.error("Error accessing webcam:", error);
        });
}

// Start webcams for both sections when page loads
window.onload = function () {
    startWebcam('registerCam');
    startWebcam('verifyCam');
};

// Capture image from webcam (for future processing)
function captureImage(videoId) {
    const video = document.getElementById(videoId);
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/png');
    
    // Display the captured image in console (for debugging)
    console.log("Captured Image:", imageData);

    // TODO: Send image data to backend for processing (if needed)
}
let streams = {}; // Store active camera streams

// Function to toggle webcam stream
function toggleCamera(videoId) {
    const video = document.getElementById(videoId);

    if (!streams[videoId]) {
        // Start webcam if it's not already active
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.classList.remove("hidden");
                streams[videoId] = stream;
            })
            .catch(error => {
                console.error("Error accessing webcam:", error);
            });
    } else {
        // Stop webcam if it's already active
        stopCamera(videoId);
    }
}

// Function to stop the webcam stream
function stopCamera(videoId) {
    if (streams[videoId]) {
        let tracks = streams[videoId].getTracks();
        tracks.forEach(track => track.stop());
        document.getElementById(videoId).classList.add("hidden");
        delete streams[videoId];
    }
}
