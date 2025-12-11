// use ES6 style import syntax (recommended)
//import * as ort from 'onnxruntime-web';
//import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const predictionText = document.getElementById("prediction");

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = "white";
ctx.lineWidth = 20;
ctx.lineCap = "round";

let drawing = false;

function startDrawing(e) {
  drawing = true;
  draw(e);
}

function stopDrawing() {
  drawing = false;
  ctx.beginPath();
}

function draw(e) {
  if (!drawing) return;

  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

clearBtn.addEventListener("click", () => {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  predictionText.textContent = "Résultat : ";
});

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mousemove", draw);

let session = null;
let modelReady = false;

async function loadModel() {
  try {
    session = await ort.InferenceSession.create("image_classifier_model.onnx");
    modelReady = true;
    console.log("Modèle chargé !");
  } catch (e) {
    console.error("Erreur chargement modèle:", e);
  }
}

loadModel();
console.log("Session: "+session)
console.log("ModelReady ? "+modelReady)

function preprocess(imageData) {
  const data = imageData.data;
  const gray = [];

  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) {
      const i = ((y * 10) * 320 + x * 10) * 4;
      gray.push((255 - data[i]) / 255); // inversion: blanc -> 1
    }
  }

  return new Float32Array(gray);
}

async function predict() {
  if (!modelReady) {
    alert("Le modèle n'est pas encore chargé !");
    return;
  }

  const inputTensor = new ort.Tensor(
    "float32",
    preprocess(ctx.getImageData(0, 0, 320, 320)),
    [1, 1, 32, 32]
  );

  const output = await session.run({ x: inputTensor });

  const scores = output[Object.keys(output)[0]].data;
  const predicted = scores.indexOf(Math.max(...scores));

  predictionText.textContent = `Résultat : ${predicted}`;
}

predictBtn.addEventListener("click", predict);
