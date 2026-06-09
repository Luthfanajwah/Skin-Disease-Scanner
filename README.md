🎯 Skin Disease Scanner — combining computer vision and explainability for real‑time skin analysis.

Skin Disease Scanner adalah aplikasi web real‑time yang digunakan untuk mengklasifikasikan berbagai jenis penyakit kulit berdasarkan gambar yang diunggah atau ditangkap langsung melalui kamera. Aplikasi ini memanfaatkan teknologi deep learning dan saliency map untuk memberikan interpretasi visual mengenai area kulit yang menjadi fokus model dalam melakukan klasifikasi.

Dibangun dengan Flask dan PyTorch aplikasi ini mendukung perbandingan multi‑model, validasi label, dan analisis fokus model (heatmap) untuk membantu mengidentifikasi indikasi penyakit kulit secara interaktif.

---

🖥️ Fitur Utama

| Fitur | Deskripsi |
| :--- | :--- |
| 📤 Upload Gambar | Prediksi dua model sekaligus, bandingkan hasil klasifikasi penyakit kulit secara berdampingan. |
| 📸 Live Camera | Tangkap gambar kondisi kulit dari webcam, langsung dapatkan hasil prediksi secara real‑time. |
| 🔬 Advanced Result (Saliency Map) | GRADCAM heatmap 
| ✅ Expected Label | Jika diagnosis/label sebenarnya diketahui, makan akan ada persentase besar kemungkinan prediksi yang benar |

---
🧠 Model yang Tersedia

*(On prosess input - coming soon)*

Aplikasi secara otomatis memuat semua model dari folder `models/`. 
*(On prosess input - coming soon)*

| Nama File | Arsitektur | Dataset | Preprocessing |
| :--- | :--- | :--- | :--- |
| model_penyakit_v1.pth ⭐ | Contoh: MobileNetV2 | Dataset V1 | Contoh: Resize + Augmentasi |
| model_penyakit_v2.pth | Contoh: ResNet50 | Dataset V2 | Contoh: CLAHE + Segmentasi |
| [Nama File Model Anda].pth | ... | ... | ... |

⭐ = Model dengan akurasi tertinggi / model unggulan.

---

🏗️ Struktur Proyek

```text
Skin-Disease-Scanner/
├── app.py                  # Flask server + PyTorch inference
├── models/                 # File bobot model (.pth)
│   ├── model_penyakit_v1.pth
│   └── ...
├── templates/
│   └── index.html          # UI utama (Tailwind CSS + JavaScript)
└── README.md
```

---

> **Catatan:** Script training (`pipeline_preprocess_dl.py`) dan dataset **tidak** disertakan dalam repositori ini.  
> Repositori ini hanya berisi aplikasi web siap‑pakai.

---

## 🚀 Cara Menjalankan

### 1. Clone Repositori
```bash
git clone https://github.com/Luthfanajwah/Undertone_Scanner.git
cd Undertone_Scanner/web_app
```

### 2. Install Dependensi
```bash
pip install flask torch torchvision pillow opencv-python matplotlib scikit-learn
```

### 3. Jalankan Server
```bash
python app.py
```

Buka browser ke **http://127.0.0.1:5050** (atau alamat IP lokal yang muncul di terminal).

Aplikasi akan otomatis memuat semua model `.pth` dari folder `models/`.

---

## 📊 Performa Model (Semua Eksperimen)

Berikut hasil evaluasi lengkap pada **test set** masing‑masing versi dataset.

| Model | Dataset | Preprocessing | Test Acc | Cool F1 | Neutral F1 | Warm F1 | Waktu Training |
|-------|---------|---------------|----------|---------|------------|---------|----------------|
*(coming soon)*


⭐ **Model unggulan:** `Mobile Net`  
- Akurasi tertinggi () dengan waktu training tercepat (  
- F1 Warm tertinggi ()  

### 🔍 Temuan Utama

---

## 🔧 API Endpoints

| Method | Route | Deskripsi |
|--------|-------|-----------|
| `GET` | `/` | Halaman utama |
| `GET` | `/get_models` | Mendapatkan daftar model yang tersedia |
| `POST` | `/predict_upload` | Prediksi gambar yang di‑upload (multipart form) |
| `POST` | `/predict_frame` | Prediksi frame kamera (base64 JSON) |
| `GET` | `/video_feed` | Streaming webcam dengan overlay prediksi |

---

## 🛠️ Teknologi yang Digunakan

- **Backend:** Flask, PyTorch, Torchvision, OpenCV, PIL
- **Frontend:** Material Symbols, Vanilla JavaScript
- **Model:** MobileNetV2
- **Visualisasi:** Matplotlib (colormap), 
