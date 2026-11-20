# Model Bahasa N-Gram — Wikipedia Bahasa Indonesia

Eksperimen pembangunan dan evaluasi model bahasa **N-Gram** pada korpus Wikipedia bahasa Indonesia. Proyek ini membandingkan pengaruh **orde N** (bigram, trigram, 4-gram) dan tiga teknik **smoothing** (Laplace, Add-k, dan Kneser-Ney) menggunakan metrik **perplexity (PPL)** serta uji prediksi kata berikutnya.

## Hasil Utama

PPL lebih rendah lebih baik. Evaluasi dilakukan pada 500 baris data uji.

| Orde N | Laplace | Add-k | Kneser-Ney | Model Terbaik |
|---:|---:|---:|---:|---|
| N=2 | 40.036,65 | 4.947,80 | **1.042,72** | **KN** |
| N=3 | 158.628,63 | 43.225,03 | **1.622,97** | **KN** |
| N=4 | 229.838,58 | 111.013,48 | **2.826,02** | **KN** |

**Kesimpulan singkat:**

- Model terbaik adalah **N=2 dengan Kneser-Ney (PPL 1.042,72)**.
- Kneser-Ney konsisten menjadi smoothing terbaik pada semua orde.
- Orde yang lebih tinggi tidak selalu lebih baik karena sparsitas data yang ekstrem dan implementasi backoff KN yang disederhanakan (langsung ke unigram kontinuasi).

Penjelasan lengkap ada di [`docs/analisis-teknis.md`](docs/analisis-teknis.md) dan [`docs/laporan.md`](docs/laporan.md).

## Struktur Proyek

```text
.
├── 01-clean-corpus.py           # Parsing dump Wikipedia, cleaning wikitext, split train/test
├── 02_run_experiments.py        # Kelas model N-Gram, training, simpan model .pkl
├── 03_generate_report_table.py  # Evaluasi perplexity dan tabel hasil
├── 04_inference_demo.py         # Demo interaktif: PPL kalimat + top-5 prediksi kata
├── docs/
│   ├── laporan.md               # Laporan tugas
│   └── analisis-teknis.md       # Analisis teknis dan pembahasan hasil
├── results/
│   └── perplexity.csv           # Ringkasan hasil perplexity
├── requirements.txt
└── README.md
```

## Kebutuhan

- **Python 3.9+**
- Tidak ada dependensi eksternal — seluruh program hanya memakai standard library Python.

```bash
python --version
```

## Cara Menjalankan

> Semua perintah dijalankan dari **root direktori proyek** karena skrip menggunakan path relatif.

### 1. Siapkan korpus

Unduh dump Wikipedia bahasa Indonesia, lalu letakkan file XML di root proyek:

```text
idwiki-latest-pages-articles.xml
```

Sumber: <https://dumps.wikimedia.org/idwiki/latest/>

### 2. Bersihkan korpus dan split data

```bash
python 01-clean-corpus.py
```

Menghasilkan:

- `train_corpus.txt` (80%)
- `test_corpus.txt` (20%)

Konfigurasi default: sampling **5%** artikel, hanya namespace 0, redirect diabaikan, dan `random.seed(42)`.

### 3. Training model

```bash
python 02_run_experiments.py
```

Melatih 9 kombinasi model (`N = 2, 3, 4` × `Laplace, Add-k, KN`) dan menyimpannya sebagai:

```text
ngram_model_N{n}_{smoothing}.pkl
```

Training memiliki mekanisme cache: model yang file `.pkl`-nya sudah ada akan dilewati.

### 4. Evaluasi dan tabel hasil

```bash
python 03_generate_report_table.py
```

Memuat seluruh model, menghitung PPL pada 500 baris data uji, lalu mencetak tabel hasil.

### 5. Demo inferensi

```bash
python 04_inference_demo.py
```

Contoh:

```text
Masukkan kalimat: jakarta adalah ibu kota indonesia

[+] Hasil Perplexity Kalimat : 124.34
[+] Latensi Komputasi        : 196.93 ms
[+] Top-5 Prediksi Kata Berikutnya:
    1. yang              (Prob: 0.0463)
    2. dan               (Prob: 0.0242)
    3. pada              (Prob: 0.0163)
    4. di                (Prob: 0.0127)
    5. daftar            (Prob: 0.0125)
```

Ketik `q` untuk keluar.

## Dataset dan Artefak

File berikut **tidak disertakan** di repositori karena ukurannya sangat besar:

| File | Ukuran |
|---|---:|
| `idwiki-latest-pages-articles.xml` | ±5,93 GB |
| `train_corpus.txt` / `test_corpus.txt` | ±61,6 MB / ±15,1 MB |
| 9 file model `.pkl` | ±1,92 GiB total |

Semua file tersebut sudah masuk `.gitignore` dan dapat dibuat ulang dengan menjalankan pipeline di atas.

## Metodologi Singkat

- **Tokenisasi:** regex `\b\w+\b`, lowercase.
- **Pemecah kalimat:** regex `[.!?]+`.
- **Padding:** `<BOS>` sebanyak `n-1` token dan `<EOS>` di akhir.
- **Smoothing:** Laplace, Add-k (`k=0,01`), dan interpolated absolute discounting ala Kneser-Ney (`d=0,75`).
- **Evaluasi:** perplexity basis 2 pada 500 baris data uji.
- **Validasi tambahan:** evaluasi pada subset token yang sama untuk memastikan perbedaan antar orde bukan artefak jumlah token.

## Catatan

- Implementasi Kneser-Ney di proyek ini adalah **penyederhanaan**, bukan KN rekursif penuh: backoff langsung menuju distribusi unigram kontinuasi, bukan bigram.
- Model belum menangani OOV dengan token `<UNK>`.
- PPL absolut masih tinggi, sehingga model ini diposisikan sebagai **eksperimen komparatif**, bukan model siap produksi.
- Repositori ini dibuat untuk keperluan tugas mata kuliah NLP.

## Penulis

**Muhammad Rizkiana Pratama** — NIM 2404421
