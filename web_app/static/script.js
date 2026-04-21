let currentMode = 'upload';
let stream = null;

// Switch tab
function switchTab(mode) {
    currentMode = mode;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));

    if (mode === 'upload') {
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
        document.getElementById('upload-tab').classList.add('active');
        stopCamera();
    } else {
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
        document.getElementById('live-tab').classList.add('active');
        startCamera();
    }
}

// Kamera
async function startCamera() {
    const video = document.getElementById('video');
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        video.srcObject = stream;
    } catch (err) {
        alert('Cannot access camera: ' + err.message);
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

// Upload & Preview
const fileInput = document.getElementById('file-input');
const preview = document.getElementById('preview');
const previewContainer = document.getElementById('preview-container');
const predictBtn = document.getElementById('predict-btn');

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            preview.src = event.target.result;
            previewContainer.style.display = 'block';
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
});

predictBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    predictBtn.disabled = true;
    predictBtn.textContent = 'Predicting...';

    try {
        const response = await fetch('/predict_upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        displayResult(data);
    } catch (err) {
        alert('Prediction failed: ' + err.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict';
    }
});

// Live Capture
document.getElementById('capture-btn').addEventListener('click', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/jpeg');
    predictFrame(imageData);
});

async function predictFrame(imageData) {
    const captureBtn = document.getElementById('capture-btn');
    captureBtn.disabled = true;
    captureBtn.textContent = 'Predicting...';

    try {
        const response = await fetch('/predict_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const data = await response.json();
        displayResult(data);
    } catch (err) {
        alert('Prediction failed: ' + err.message);
    } finally {
        captureBtn.disabled = false;
        captureBtn.textContent = 'Capture & Predict';
    }
}

// Tampilkan hasil
function displayResult(data) {
    const resultDiv = document.getElementById('result');
    const placeholder = document.getElementById('placeholder');
    const predClass = document.getElementById('pred-class');
    const predConf = document.getElementById('pred-conf');
    const probBars = document.getElementById('prob-bars');

    predClass.textContent = data.class;
    predConf.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

    // Buat probability bars
    probBars.innerHTML = '';
    data.probabilities.forEach((prob, idx) => {
        const percent = (prob * 100).toFixed(1);
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <span class="prob-label">${CLASS_NAMES[idx]}</span>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: ${percent}%;"></div>
            </div>
            <span class="prob-value">${percent}%</span>
        `;
        probBars.appendChild(item);
    });

    resultDiv.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';
}

// Cleanup saat halaman ditutup
window.addEventListener('beforeunload', () => {
    stopCamera();
});

let isPredicting = false;

document.getElementById('capture-btn').addEventListener('click', async () => {
    if (isPredicting) return;
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');

    // Freeze: set ukuran canvas sesuai video dan gambar frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Tampilkan canvas sebagai ganti video (freeze effect)
    video.style.display = 'none';
    canvas.style.display = 'block';

    const imageData = canvas.toDataURL('image/jpeg');
    isPredicting = true;
    document.getElementById('capture-btn').disabled = true;
    document.getElementById('capture-btn').textContent = 'Processing...';

    try {
        const response = await fetch('/predict_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const data = await response.json();
        displayResult(data);

        // Jika ingin juga menampilkan Grad‑CAM, panggil endpoint terpisah
        const gradcamResponse = await fetch('/gradcam', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const gradcamData = await gradcamResponse.json();
        if (gradcamData.heatmap) {
            displayGradcam(gradcamData.heatmap);
        }
    } catch (err) {
        alert('Prediction failed: ' + err.message);
    } finally {
        // Kembalikan tampilan video
        video.style.display = 'block';
        canvas.style.display = 'none';
        isPredicting = false;
        document.getElementById('capture-btn').disabled = false;
        document.getElementById('capture-btn').textContent = 'Capture & Predict';
    }
});

// Fungsi untuk mengubah elemen img menjadi base64 (data URL)
function getBase64FromImageElement(imgElement) {
    const canvas = document.createElement('canvas');
    canvas.width = imgElement.naturalWidth;
    canvas.height = imgElement.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0);
    return canvas.toDataURL('image/jpeg');
}

// Modifikasi event listener predictBtn
predictBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    predictBtn.disabled = true;
    predictBtn.textContent = 'Predicting...';

    try {
        // 1. Prediksi kelas
        const response = await fetch('/predict_upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        displayResult(data);

        // 2. Dapatkan base64 dari gambar preview
        const previewImg = document.getElementById('preview');
        const imageData = getBase64FromImageElement(previewImg);

        // 3. Minta Grad‑CAM
        const gradcamResponse = await fetch('/gradcam', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const gradcamData = await gradcamResponse.json();
        if (gradcamData.heatmap) {
            displayGradcam(gradcamData.heatmap);
        }
    } catch (err) {
        alert('Prediction failed: ' + err.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict';
    }
});

// Fungsi untuk menampilkan heatmap Grad‑CAM
function displayGradcam(heatmapBase64) {
    const heatmapImg = document.getElementById('gradcam-img');
    heatmapImg.src = 'data:image/jpeg;base64,' + heatmapBase64;
    document.getElementById('gradcam-container').style.display = 'block';
}