const video = document.querySelector("#camera");
const statusText = document.querySelector("#status");
const objectName = document.querySelector("#resultname");
const confidence = document.querySelector("#resultaccuracy");
let classifier;
let previousLabel = "";
let stopped = false;

function stop(message) {
  stopped = true;
  statusText.textContent = message;
  objectName.textContent = confidence.textContent = "—";
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
  }
}

const frame = document.createElement("canvas");
frame.width = frame.height = 224;
const context = frame.getContext("2d", { willReadFrequently: true });
const pixelCount = frame.width * frame.height;
const input = new Float32Array(3 * pixelCount);

async function classifyFrame() {
  if (stopped) return;
  try {
    const size = Math.min(video.videoWidth, video.videoHeight);
    context.drawImage(video, (video.videoWidth - size) / 2, (video.videoHeight - size) / 2,
      size, size, 0, 0, frame.width, frame.height);
    const pixels = context.getImageData(0, 0, frame.width, frame.height).data;
    for (let i = 0; i < pixelCount; i++) {
      input[i] = (pixels[4 * i] / 255 - 0.485) / 0.229;
      input[pixelCount + i] = (pixels[4 * i + 1] / 255 - 0.456) / 0.224;
      input[2 * pixelCount + i] = (pixels[4 * i + 2] / 255 - 0.406) / 0.225;
    }
    const tensor = new ort.Tensor("float32", input, [1, 3, 224, 224]);
    const outputs = await classifier.run({ [classifier.inputNames[0]]: tensor });
    if (stopped) return;
    const scores = outputs[classifier.outputNames[0]].data;
    let best = 0;
    for (let i = 1; i < scores.length; i++) {
      if (scores[i] > scores[best]) best = i;
    }
    let sum = 0;
    for (const score of scores) sum += Math.exp(score - scores[best]);
    const prediction = { label: imageLabels[best], confidence: 1 / sum };
    objectName.textContent = prediction.label;
    confidence.textContent = (100 * prediction.confidence).toFixed(1) + "%";
    statusText.textContent = prediction.confidence > 0.5 ? "" : "Low confidence. Try centering the object.";
    if (prediction.confidence > 0.5) {
      if (prediction.label !== previousLabel && window.speechSynthesis && window.SpeechSynthesisUtterance) {
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(new SpeechSynthesisUtterance("Object detected is " + prediction.label));
      }
      previousLabel = prediction.label;
    } else {
      previousLabel = "";
    }
  } catch (error) {
    console.error(error);
    statusText.textContent = "Could not classify the camera frame: " + error.message;
    return;
  }
  setTimeout(classifyFrame, 200);
}

async function start() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    stop("Open this page on localhost or HTTPS in a browser with camera support.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    stream.getVideoTracks()[0].addEventListener("ended", () => stop("Camera disconnected. Reconnect it and reload."));
    await video.play();
  } catch (error) {
    stop(error.name === "NotAllowedError"
      ? "Camera access was denied. Allow camera access and reload."
      : "Could not start the camera. Check that it is connected and available, then reload.");
    return;
  }

  if (stopped) return;
  statusText.textContent = "Loading SqueezeNet…";
  try {
    if (!window.ort || typeof modelWeights === "undefined") {
      throw new Error("The model files are missing. Extract the full ZIP and reopen index.html.");
    }
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = false;
    ort.env.wasm.wasmPaths = { "ort-wasm.wasm": wasmURL };
    const bytes = Uint8Array.from(atob(modelWeights), ch => ch.charCodeAt(0));
    classifier = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    if (stopped) return;
    statusText.textContent = "";
    classifyFrame();
  } catch (error) {
    console.error(error);
    if (!stopped) statusText.textContent = "SqueezeNet could not start: " + error.message;
  }
}

start();
