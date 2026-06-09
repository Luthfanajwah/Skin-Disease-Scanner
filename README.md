🎯 Skin Disease Scanner — combining computer vision and explainability for real‑time skin analysis.

Skin Disease Scanner adalah aplikasi web real‑time yang digunakan untuk mengklasifikasikan berbagai jenis penyakit kulit berdasarkan gambar yang diunggah atau ditangkap langsung melalui kamera. Aplikasi ini memanfaatkan teknologi deep learning dan saliency map untuk memberikan interpretasi visual mengenai area kulit yang menjadi fokus model dalam melakukan klasifikasi.

Dibangun dengan Flask, PyTorch, dan Tailwind CSS, aplikasi ini mendukung perbandingan multi‑model, validasi label, dan analisis fokus model (ROI bounding box + heatmap) untuk membantu mengidentifikasi indikasi penyakit kulit secara interaktif.

🖥️ Fitur Utama

| Fitur | Deskripsi |
| :--- | :--- |
| 📤 Upload Gambar | Prediksi dua model sekaligus, bandingkan hasil klasifikasi penyakit kulit secara berdampingan. |
| 📸 Live Camera | Tangkap gambar kondisi kulit dari webcam, langsung dapatkan hasil prediksi secara real‑time. |
| 📚 Bulk Upload | Proses puluhan gambar sekaligus, lihat hasil dalam grid beserta statistik akurasi per model. |
| 🧠 Multi‑Model | Muat otomatis semua model `.pth` dalam folder `models/`, pilih dua model penyakit kulit untuk dibandingkan. |
| 🔬 Advanced Result (Saliency Map) | Lihat heatmap fokus model pada lesi/gejala kulit, bounding box ROI, dan penjelasan; kontrol interaktif untuk threshold, colormap, dan warna kotak. |
| ✅ Expected Label | Jika diagnosis/label sebenarnya diketahui, hasil prediksi akan di‑highlight hijau (benar) atau merah (salah). |
| ⚖️ Statistik Akurasi Bulk | Menghitung persentase benar/salah Model A vs Model B saat *expected label* diisi. |

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

Cara Menjalankan:
1. Clone RepositoriBashgit clone [https://github.com/Luthfanajwah/Skin-Disease-Scanner.git](https://github.com/Luthfanajwah/Skin-Disease-Scanner.git)
   
```cd Skin-Disease-Scanner```
*(Catatan: Sesuaikan perintah cd di atas jika folder aplikasi web berada di dalam sub-folder tertentu, misalnya cd Skin-Disease-Scanner/web_app)*
2. Install Dependensi
```Bashpip install flask torch torchvision pillow opencv-python matplotlib scikit-learn```
3. Jalankan Server
```Bashpython app.py```
4. Buka browser dan akses ke 
```http://127.0.0.1:5050``` (atau alamat IP lokal yang muncul di terminal Anda). 

📊 Performa Model
*(On prosess input - coming soon)*

ModelDatasetPreprocessingTest AccPenyakit A (F1)Penyakit B (F1)Waktu TrainingModel A ⭐V1Preprocess 100.00%0.000.0000 menitModel BV1Preprocess 200.00%0.000.0000 menit

⭐ Model unggulan: [Isi Nama Model Terbaik.pth]Akurasi tertinggi (00.00%) dengan performa deteksi lesi kulit paling stabil.Arsitektur efisien dan cocok untuk deployment real-time.

🔍 Temuan Utama (Template - Silakan Sesuaikan)Isi dengan analisis perbandingan antar model Anda di sini.Contoh: Arsitektur X lebih sensitif terhadap tekstur kulit, sedangkan arsitektur Y lebih baik dalam mendeteksi perubahan warna.Contoh: Preprocessing tertentu membantu model mengabaikan rambut halus pada sampel kulit.

🔧 API EndpointsMethodRouteDeskripsiGET/Halaman utama aplikasi web.GET/get_modelsMendapatkan daftar model penyakit kulit yang tersedia di folder models/.POST/predict_uploadPrediksi gambar kulit yang di‑upload (multipart form).POST/predict_framePrediksi frame kamera real-time (base64 JSON).POST/predict_bulkPrediksi banyak gambar sampel kulit sekaligus.POST/saliencyMenghasilkan heatmap saliency map + ROI pada area kulit yang terindikasi.GET/video_feedStreaming webcam dengan overlay prediksi penyakit kulit.

🛠️ Teknologi yang DigunakanBackend
Flask, PyTorch, Torchvision, OpenCV, PILFrontend: Tailwind CSS (CDN), Material Symbols, Vanilla JavaScriptModel 
Klasifikasi: Deep Learning / Transfer Learning (PyTorch)
Visualisasi & Interpretabilitas: Matplotlib (Colormap), Saliency Map, Bounding Box ROI
