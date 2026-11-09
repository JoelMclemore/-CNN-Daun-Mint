# Klasifikasi Daun Mint Segar dan Kering Menggunakan CNN

Aplikasi klasifikasi citra daun mint untuk membedakan dua kondisi daun, yaitu **Fresh (segar)** dan **Dried (kering)**. Model dibangun menggunakan **Convolutional Neural Network (CNN)** dengan TensorFlow/Keras, kemudian diintegrasikan ke aplikasi web menggunakan Streamlit.

## Demo Aplikasi

Aplikasi dapat dicoba melalui tautan berikut:

**[Buka aplikasi klasifikasi daun mint di Streamlit](https://jtapp88mfaad8rnnyavvwng.streamlit.app/)**

## Tujuan Proyek

1. Menerapkan pengolahan citra dan deep learning untuk klasifikasi kondisi daun mint.
2. Membangun model CNN untuk membedakan citra daun mint segar dan kering.
3. Membandingkan beberapa konfigurasi CNN dan memilih model berdasarkan hasil validasi.
4. Mengimplementasikan model terpilih agar dapat digunakan melalui aplikasi berbasis web.

## Dataset

Dataset yang digunakan adalah **Pudina Leaf Dataset: Freshness Analysis** dari Mendeley Data.

- Sumber dataset: [Mendeley Data - Pudina Leaf Dataset: Freshness Analysis](https://doi.org/10.17632/nvbpydc3fs.1)
- Kelas yang digunakan: `Fresh` dan `Dried`.
- Citra `Dried/Multiple` sebanyak 132 gambar dikeluarkan agar penelitian berfokus pada satu daun dalam satu gambar.
- Jumlah data akhir: **3.522 citra**, terdiri dari 1.773 citra Fresh dan 1.749 citra Dried/Single.

Kelas `Spoiled` dari dataset sumber tidak digunakan dalam penelitian ini.

## Metodologi

Tahapan utama penelitian adalah:

1. **Eksplorasi dan seleksi data** - memeriksa struktur folder, jumlah gambar, format, ukuran, dan keterbacaan citra.
2. **Pemeriksaan duplikasi** - menggunakan MD5 untuk mendeteksi file yang identik dan perceptual hash (dHash) untuk mengidentifikasi citra yang memiliki kemiripan visual.
3. **Preprocessing** - mengubah ukuran citra menjadi 128 × 128 piksel dalam format RGB dan menormalisasi nilai piksel dengan membaginya dengan 255 sehingga berada pada rentang 0–1.
4. **Pembagian dataset** - membagi data secara berkelompok dan berstrata menjadi data training, validation, dan testing. Pengelompokan digunakan untuk mengurangi risiko citra yang sangat mirip tersebar ke subset berbeda.
5. **Pelatihan CNN** - menjalankan tiga skenario eksperimen.
6. **Evaluasi dan pemilihan model** - memilih model berdasarkan validation loss terendah, kemudian mengevaluasinya pada data test.
7. **Deployment** - menggunakan model terpilih pada aplikasi web Streamlit.

### Pembagian Dataset

| Subset | Dried | Fresh | Total | Persentase |
|---|---:|---:|---:|---:|
| Training | 1.229 | 1.237 | 2.466 | 70,0% |
| Validation | 261 | 271 | 532 | 15,1% |
| Testing | 259 | 265 | 524 | 14,9% |
| **Total** | **1.749** | **1.773** | **3.522** | **100%** |

## Arsitektur Model CNN

Model CNN dasar terdiri dari tiga blok konvolusi. Setiap blok menggunakan Conv2D dengan kernel 3 × 3 dan aktivasi ReLU, diikuti MaxPooling2D. Jumlah filter yang digunakan berturut-turut adalah 16, 32, dan 64.

| Tahap | Konfigurasi |
|---|---|
| Input | 128 × 128 × 3 (RGB) |
| Blok konvolusi 1 | Conv2D, 16 filter, kernel 3 × 3, ReLU + MaxPooling2D |
| Blok konvolusi 2 | Conv2D, 32 filter, kernel 3 × 3, ReLU + MaxPooling2D |
| Blok konvolusi 3 | Conv2D, 64 filter, kernel 3 × 3, ReLU + MaxPooling2D |
| Ekstraksi fitur | GlobalAveragePooling2D |
| Fully connected | Dense 64 neuron, ReLU |
| Output | Dense 1 neuron, Sigmoid |

Model CNN dasar memiliki **27.809 parameter**. Output sigmoid digunakan untuk menghasilkan skor kelas positif `Fresh`, dengan ambang keputusan 0,5:

- Skor ≥ 0,5: `Fresh` (segar)
- Skor < 0,5: `Dried` (kering)

## Skenario Eksperimen

Penelitian membandingkan tiga konfigurasi:

- **Eksperimen A - CNN dasar:** tanpa augmentasi, Batch Normalization, dan Dropout.
- **Eksperimen B - CNN + augmentasi:** menggunakan augmentasi pada data training berupa RandomFlip, RandomRotation, RandomZoom, dan RandomBrightness.
- **Eksperimen C - CNN + optimasi tambahan:** menggunakan augmentasi, Batch Normalization, Dropout 0,3, dan ReduceLROnPlateau.

Konfigurasi pelatihan utama menggunakan Adam dengan learning rate awal 0,001, Binary Crossentropy, batch size 32, dan maksimum 30 epoch. EarlyStopping dan ModelCheckpoint memantau validation loss.

## Hasil Evaluasi

Model terbaik adalah **Eksperimen A (CNN dasar)** karena memperoleh validation loss terendah, yaitu 0,0307.

Hasil pengujian akhir pada 524 citra data test:

| Metrik | Hasil |
|---|---:|
| Accuracy | **98,85%** |
| Precision (kelas positif: Fresh) | **98,87%** |
| Recall (kelas positif: Fresh) | **98,87%** |
| F1-score (kelas positif: Fresh) | **98,87%** |

Confusion matrix menunjukkan 518 prediksi benar dan 6 prediksi salah:

| Label sebenarnya | Prediksi Dried | Prediksi Fresh |
|---|---:|---:|
| Dried | 256 | 3 |
| Fresh | 3 | 262 |

Nilai evaluasi tersebut berasal dari pengujian pada dataset yang digunakan dalam penelitian. Pengujian menggunakan foto eksternal dengan kondisi pencahayaan, latar belakang, dan perangkat pengambilan gambar yang lebih beragam masih diperlukan.

## Teknologi yang Digunakan

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Pillow (PIL)
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm
- tf.data

Versi yang tercatat pada lingkungan eksperimen notebook adalah TensorFlow 2.21.0 dan Scikit-learn 1.6.1. Versi library pada lingkungan deployment dapat berbeda mengikuti konfigurasi `requirements.txt` di repository.

## Menjalankan Proyek Secara Lokal

Langkah berikut merupakan contoh untuk menjalankan aplikasi Streamlit dari source code. Perintah ini mengasumsikan repository memiliki `requirements.txt` dan file utama aplikasi bernama `app.py`; sesuaikan nama file jika berbeda.

### 1. Clone repository

```bash
git clone https://github.com/JoelMclemore/-CNN-Daun-Mint.git
cd ./-CNN-Daun-Mint
```

### 2. Buat dan aktifkan virtual environment (opsional, disarankan)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

Setelah perintah dijalankan, buka alamat lokal yang ditampilkan oleh Streamlit pada terminal.

## Notebook Penelitian

Notebook `CNN_Daun_Mint_Segar.ipynb` berisi proses eksperimen, mulai dari eksplorasi dataset, pemeriksaan kualitas dan kemiripan citra, preprocessing, pembagian data, pembangunan model CNN, pelatihan tiga eksperimen, hingga evaluasi model.

Notebook penelitian dijalankan menggunakan Google Colab. Untuk menjalankan ulang seluruh proses training, siapkan file ZIP dataset Pudina yang memiliki folder `Fresh` dan `Dried`, kemudian ikuti sel notebook secara berurutan.

## Tautan Proyek

- **GitHub repository:** [JoelMclemore/-CNN-Daun-Mint](https://github.com/JoelMclemore/-CNN-Daun-Mint)
- **Aplikasi Streamlit:** [Klasifikasi Daun Mint](https://jtapp88mfaad8rnnyavvwng.streamlit.app/)
- **Notebook Google Colab:** [Buka notebook penelitian](https://colab.research.google.com/drive/19DEm_rvqxPvnnzBJb9Pxp7lfabpeKeC6)
- **Video demo:** [Folder video demo proyek](https://mikroskilacid-my.sharepoint.com/:f:/g/personal/luthfi_aulia_students_mikroskil_ac_id/IgAgs4S-Y8I4Q4Jt-luzeRSuAcNmmEJ0sUqxIY7f7zgMN6M?e=w6barQ)

## Anggota Kelompok

1. **Joel Saroha Sianturi** - pengembangan arsitektur CNN, pelatihan eksperimen A, B, dan C, evaluasi selama pelatihan, serta perbandingan dan pemilihan model terbaik.
2. **Josua Christian Dachi** - persiapan lingkungan, pengunggahan dataset, eksplorasi struktur dan sampel gambar, pemeriksaan integritas, serta analisis distribusi dan ukuran gambar.
3. **Alvin Charles Revaldo Matondang** - pemeriksaan duplikasi, pengelompokan gambar serupa, penyaringan data, analisis karakteristik gambar, pembagian dataset, preprocessing, dan augmentasi.
4. **Lutfi Aulia Tarswa** - evaluasi akhir pada data uji, analisis kesalahan klasifikasi, penyimpanan model, demonstrasi prediksi gambar, dan pengemasan hasil proyek.

