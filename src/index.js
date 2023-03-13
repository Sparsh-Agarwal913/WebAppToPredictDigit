// import * as tf from "@tensorflow/tfjs";

const canvas = document.getElementById("myCanvas");
const context = canvas.getContext("2d");

canvas.style.border = "1px solid black";
context.fillStyle = "#000"; // set fill color to black
context.strokeStyle = "#000"; // set stroke color to black
context.lineWidth = 10; // set line width to 5 pixels
context.imageSmoothingEnabled = false; // disable anti-aliasing

let isDrawing = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener("mousedown", (event) => {
  isDrawing = true;
  lastX = event.clientX - canvas.offsetLeft;
  lastY = event.clientY - canvas.offsetTop;
});

canvas.addEventListener("mousemove", (event) => {
  if (isDrawing) {
    const currentX = event.clientX - canvas.offsetLeft;
    const currentY = event.clientY - canvas.offsetTop;

    context.beginPath();
    context.moveTo(lastX, lastY);
    context.lineTo(currentX, currentY);
    context.stroke();

    lastX = currentX;
    lastY = currentY;
  }
});

canvas.addEventListener("mouseup", () => {
  isDrawing = false;
});

function saveCanvas() {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;

  const dataURL = canvas.toDataURL("image/png");
  const img = new Image();
  img.onload = () => {
    const ctx = tempCanvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const tempDataURL = tempCanvas.toDataURL("image/png");
    const tensorImg = tf.browser.fromPixels(tempCanvas);

    const resizedImg = tf.image.resizeBilinear(tensorImg, [28, 28]);
    const normalizedImg = resizedImg.div(255.0);
    const batchedImg = normalizedImg.expandDims(0);
    modelprediction(batchedImg);
    // do something with the tensor
  };
  img.src = dataURL;
}

const modelJSON = module.require("../public/model.json");
const modelWeights = module.require("../public/group1-shard1of1.bin");

async function modelprediction(finalimage) {
  const model = await tf.loadLayersModel(
    bundleResourceIO(modelJSON, modelWeights)
  );
  const prediction = await model.predict(finalimage);
  console.log(prediction);
}

const predictButton = document.getElementById("save_btn");
if (predictButton) {
  predictButton.onclick = function () {
    saveCanvas();
  };
}
