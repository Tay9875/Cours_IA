// use ES6 style import syntax (recommended)
//import * as ort from 'onnxruntime-web';
//import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const scoresText = document.getElementById("scores");
const outputText = document.getElementById("output");
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
  scoresText.textContent = "Scores : ";
  outputText.textContent = "Output : ";
});

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mousemove", draw);

let session = null;
let modelReady = false;
let inputName = null;

async function loadModel() {
  try {
    session = await ort.InferenceSession.create("image_classifier_tensorboard.onnx");
    //await ort.InferenceSession.create("image_classifier_cnn_dropout.onnx");
    //await ort.InferenceSession.create("image_classifier_cnn.onnx");
    //await ort.InferenceSession.create("image_classifier_model.onnx");

    modelReady = true;
    
    // Récupérer le nom exact de l'input
    inputName = session.inputNames[0];
    
    console.log("Modèle chargé !");
    console.log("Input names:", session.inputNames);
    console.log("Output names:", session.outputNames);
    console.log("Input name utilisé:", inputName);
  } catch (e) {
    console.error("Erreur chargement modèle:", e);
  }
}

loadModel();

function preprocess(imageData) {
  const smallCanvas = document.createElement("canvas");
  smallCanvas.width = 32;
  smallCanvas.height = 32;
  const smallCtx = smallCanvas.getContext("2d");
  
  // Redimensionner avec interpolation de bonne qualité
  smallCtx.imageSmoothingEnabled = true;
  smallCtx.imageSmoothingQuality = 'high';
  smallCtx.drawImage(canvas, 0, 0, 32, 32);
  
  const smallData = smallCtx.getImageData(0, 0, 32, 32).data;
  const gray = new Float32Array(32 * 32);
  
  // Convertir en niveaux de gris et normaliser comme ToTensor() dans PyTorch
  for (let i = 0; i < 32 * 32; i++) {
    const r = smallData[i * 4];
    const g = smallData[i * 4 + 1];
    const b = smallData[i * 4 + 2];
    
    // Moyenne des 3 canaux pour niveau de gris
    const grayValue = (r + g + b) / 3;
    
    // INVERSION + Normalisation [0, 1] comme ToTensor()
    // Fond noir (0) -> 0, Chiffre blanc (255) -> 1
    gray[i] = (255 - grayValue) / 255.0;
  }
  
  console.log("Gray min/max:", Math.min(...gray), Math.max(...gray));
  console.log("Gray sample:", Array.from(gray.slice(0, 10)));
  console.log("Gray mean:", gray.reduce((a, b) => a + b, 0) / gray.length);
  
  return gray;
}

async function predict() {
  if (!modelReady) {
    alert("Le modèle n'est pas encore chargé !");
    return;
  }
  
  try {
    const processedData = preprocess(ctx.getImageData(0, 0, 320, 320));
    
    // Créer le tensor avec le nom d'input correct
    const inputTensor = new ort.Tensor(
      "float32",
      processedData,
      [1, 1, 32, 32]
    );
    
    console.log("Input tensor shape:", inputTensor.dims);
    console.log("Input tensor data sample:", inputTensor.data.slice(0, 10));
    
    // Utiliser le nom d'input dynamique
    const feeds = {};
    feeds[inputName] = inputTensor;
    
    const output = await session.run(feeds);
    
    console.log("Output keys:", Object.keys(output));
    
    // Récupérer les scores (première clé de l'output)
    const outputTensor = output[session.outputNames[0]];
    const scores = Array.from(outputTensor.data);
    
    console.log("Scores bruts:", scores);
    
    // Appliquer softmax pour avoir des probabilités
    const expScores = scores.map(s => Math.exp(s));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const probabilities = expScores.map(s => s / sumExp);
    
    console.log("Probabilités:", probabilities);
    
    // Trouver la classe prédite
    const predicted = probabilities.indexOf(Math.max(...probabilities));
    const confidence = (probabilities[predicted] * 100).toFixed(2);
    
    // Afficher les résultats
    outputText.textContent = `Output shape : ${outputTensor.dims}`;
    scoresText.textContent = `Scores : ${probabilities.map(p => (p * 100).toFixed(1) + '%').join(', ')}`;
    predictionText.textContent = `Résultat : ${predicted} (confiance: ${confidence}%)`;
    
  } catch (e) {
    console.error("Erreur lors de la prédiction:", e);
    alert("Erreur lors de la prédiction: " + e.message);
  }
}

predictBtn.addEventListener("click", predict);