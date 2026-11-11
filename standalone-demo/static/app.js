const form = document.querySelector("#predict-form");
const input = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const selectedFile = document.querySelector("#selected-file");
const fileName = document.querySelector("#file-name");
const clearFile = document.querySelector("#clear-file");
const button = document.querySelector("#predict-button");
const error = document.querySelector("#error");
const resultPanel = document.querySelector("#result-panel");
const emptyState = document.querySelector("#empty-state");
const loadingState = document.querySelector("#loading-state");
const result = document.querySelector("#result");

// This site intentionally keeps no browser-side history. Clear any data left
// by an older local build that used the same localhost origin.
localStorage.clear();
sessionStorage.clear();
if ("caches" in window) {
  caches.keys().then((names) => Promise.all(names.map((name) => caches.delete(name))));
}

form.reset();
document.querySelector("#result-image").removeAttribute("src");

function selectFile(file) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  fileName.textContent = file.name;
  selectedFile.hidden = false;
  button.disabled = false;
  error.hidden = true;
}

input.addEventListener("change", () => {
  if (input.files[0]) selectFile(input.files[0]);
});

clearFile.addEventListener("click", () => {
  input.value = "";
  selectedFile.hidden = true;
  button.disabled = true;
  error.hidden = true;
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file?.type.startsWith("image/")) selectFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!input.files[0]) return;

  error.hidden = true;
  emptyState.hidden = true;
  result.hidden = true;
  loadingState.hidden = false;
  resultPanel.classList.remove("is-empty");
  button.disabled = true;

  const body = new FormData();
  body.append("image", input.files[0]);

  try {
    const response = await fetch("/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");

    document.querySelector("#result-image").src = data.preview;
    document.querySelector("#prediction").textContent = data.prediction;
    document.querySelector("#confidence").textContent = `${Math.round(data.confidence * 100)}%`;

    const list = document.querySelector("#top-results");
    list.replaceChildren();
    data.top.forEach((item) => {
      const row = document.createElement("li");
      row.style.setProperty("--score", `${Math.max(item.confidence * 100, 2)}%`);
      const label = document.createElement("span");
      label.textContent = item.label;
      const score = document.createElement("strong");
      score.textContent = `${(item.confidence * 100).toFixed(1)}%`;
      row.append(label, score);
      list.append(row);
    });

    loadingState.hidden = true;
    result.hidden = false;
  } catch (requestError) {
    loadingState.hidden = true;
    emptyState.hidden = false;
    error.textContent = requestError.message;
    error.hidden = false;
  } finally {
    button.disabled = false;
  }
});
