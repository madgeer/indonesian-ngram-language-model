# Laporan Tugas Membuat Model N-Gram

**Nama:** Muhammad Rizkiana Pratama  
**NIM:** 2404421

---

## Deskripsi

### Pendahuluan

Pemodelan Bahasa (*Language Modeling*) merupakan salah satu teknik dasar dalam Pemrosesan Bahasa Alami (*Natural Language Processing* / NLP) yang bertujuan untuk menghitung probabilitas kemunculan suatu urutan kata atau memprediksi kata berikutnya dalam suatu kalimat. Salah satu pendekatan klasik berbasis statistik yang paling populer adalah **Model N-Gram**. Laporan ini menampilkan eksperimen dan analisis terkait model N-Gram.

### Tujuan

1. Membangun model bahasa n-gram dari korpus Wikipedia bahasa Indonesia.
2. Membandingkan pengaruh orde n (bigram, trigram, 4-gram) terhadap kualitas model.
3. Membandingkan tiga teknik smoothing: Laplace, Add-k, dan Kneser-Ney.
4. Mengevaluasi model secara kuantitatif (*perplexity*) dan kualitatif (prediksi kata berikutnya).

### Dataset

Korpus berasal dari dump `idwiki-latest-pages-articles.xml` berukuran sekitar **5,93 GB**. Pemrosesan dilakukan dengan sampling **5%** artikel.

| Properti | Data Train | Data Test |
|---|---:|---:|
| Jumlah artikel | 31.520 | 7.881 |
| Jumlah token | 8.451.690 | 2.075.431 |
| Vocabulary | 475.762 | 170.030 |
| Ukuran file | 61,6 MB | 15,1 MB |
| Proporsi | 80% | 20% |

### Pemrosesan Data

Pemrosesan data yang dilakukan:

- Hanya mengambil halaman dengan **namespace 0** (artikel utama).
- Mengabaikan halaman **redirect** (`#ALIH`, `#REDIRECT`).
- Membersihkan wikitext: komentar, `<ref>`, template `{{...}}`, pranala kategori/berkas, heading, URL, dan karakter non-alfanumerik.
- Hanya menyimpan artikel dengan panjang teks bersih **> 50 karakter**.
- Mengacak artikel dengan `random.seed(42)`, lalu membagi **80% train / 20% test**.
- Satu baris file korpus merepresentasikan satu artikel.

### Program yang Digunakan

#### 1. `01-clean-corpus.py`

- `clean_wikitext()` menghapus komentar HTML, referensi, template, pranala wiki (label dipertahankan), heading, URL, dan karakter selain `a-z`, `A-Z`, `0-9`, spasi, serta tanda baca `. , ! ?`.
- `process_and_split_corpus()` memakai `ElementTree.iterparse` agar XML besar tidak dimuat sekaligus ke memori.
- Hasil akhir ditulis ke `train_corpus.txt` dan `test_corpus.txt`.

#### 2. `02_run_experiments.py`

Terdapat class `ComprehensiveNGram` yang menyimpan:

- `ngram_counts`: frekuensi setiap n-gram.
- `context_counts`: frekuensi setiap konteks (n-1 kata).
- `context_unique_follows`: jumlah kata berbeda yang mengikuti suatu konteks.
- `continuation_counts` dan `seen_preceding_pairs`: struktur tambahan untuk Kneser-Ney.

Alur training:

1. Membaca korpus baris per baris.
2. Memecah kalimat dengan regex `[.!?]+`.
3. Tokenisasi dengan `\b\w+\b` dan mengubah ke huruf kecil.
4. Melewati kalimat dengan jumlah token `< n`.
5. Menambahkan `n-1` token `<BOS>` dan satu `<EOS>`.
6. Menghitung n-gram, konteks, dan statistik kontinuasi untuk KN.
7. Menyimpan model menggunakan `pickle` (`.pkl`).

#### 3. `03_generate_report_table.py`

Memuat setiap `.pkl`, menjalankan `evaluate_ppl()`, dan membuat laporan berupa tabel perplexity.

#### 4. `04_inference_demo.py`

Berisi program untuk menjalankan model `.pkl` dan melakukan proses inference, meliputi perhitungan perplexity kalimat dan prediksi kata berikutnya.

### Konfigurasi

| Parameter | Nilai |
|---|---|
| Orde n | 2, 3, 4 |
| Smoothing | Laplace, Add-k, Kneser-Ney |
| Nilai k (Add-k) | 0,01 |
| Nilai d (KN) | 0,75 |
| Tokenisasi | `\b\w+\b`, lowercase |
| Padding | `<BOS>` sebanyak `n-1`, diakhiri `<EOS>` |
| Evaluasi PPL | 500 baris test, panjang > 30 karakter |
| Seed | 42 |

---

## Eksperimen

### Alur Program

1. Korpus dibersihkan dan dibagi menjadi train/test 80:20.
2. Sembilan model dilatih dengan kombinasi `n ∈ {2,3,4}` dan `smoothing ∈ {laplace, add_k, kn}`.
3. Setiap model disimpan sebagai `ngram_model_N{n}_{smoothing}.pkl`.
4. Setiap model dievaluasi dengan `evaluate_ppl()` pada 500 baris test yang sama.
5. Model terbaik diuji secara kualitatif pada beberapa konteks bahasa Indonesia.

### Hasil Perplexity

Karena nilai PPL lebih rendah lebih baik, angka di bawah dihitung ulang dari 9 file `.pkl` menggunakan metode evaluasi yang sama dengan `03_generate_report_table.py`.

| Orde N | Laplace | Add-k | Kneser-Ney | Model Terbaik |
|---:|---:|---:|---:|---|
| N=2 | 40.036,65 | 4.947,80 | **1.042,72** | **KN** |
| N=3 | 158.628,63 | 43.225,03 | **1.622,97** | **KN** |
| N=4 | 229.838,58 | 111.013,48 | **2.826,02** | **KN** |

Berdasarkan tabel di atas:

- **Kneser-Ney** menghasilkan PPL terendah pada semua orde.
- **Laplace** menghasilkan PPL tertinggi pada semua orde.
- **N=2 dengan Kneser-Ney** merupakan model terbaik dengan PPL **1.042,72**.

### Uji Prediksi Kata Selanjutnya

Diambil dari model KN terbaik pada masing-masing orde. Untuk konteks multi-kata, N=2 hanya melihat kata terakhir, sedangkan N=3 melihat dua kata terakhir. Untuk input satu kata, N=3 menggunakan konteks `<BOS> + kata`, sedangkan N=2 hanya menggunakan kata tersebut.

| Konteks | N=2 KN (Top-5) | N=3 KN (Top-5) |
|---|---|---|
| `kucing` | yang (0,1015), persia (0,0765), dan (0,0458), ini (0,0424), besar (0,0293) | persia (0,3084), ini (0,1757), yang (0,0120), merupakan (0,0090), menjadi (0,0089) |
| `presiden` | republik (0,0364), dan (0,0320), amerika (0,0290), ri (0,0258), indonesia (0,0227) | juga (0,0299), dipilih (0,0289), indonesia (0,0237), amerika (0,0236), republik (0,0235) |
| `ibu kota` | ini (0,0682), di (0,0356), yang (0,0319), dan (0,0271), depok (0,0173) | kabupaten (0,1229), kecamatan (0,0791), provinsi (0,0663), negara (0,0620), dari (0,0320) |
| `pada tahun` | yang (0,0203), 2010 (0,0151), 2016 (0,0129), 2012 (0,0127), 2007 (0,0126) | 2010 (0,0184), 2009 (0,0169), yang (0,0168), 2006 (0,0167), 2007 (0,0164) |

---

## Pengujian Inference

Pengujian inference dilakukan dengan menghitung perplexity sebuah kalimat dan menampilkan top-5 prediksi kata berikutnya. Contoh pengujian pada kalimat:

> `jakarta adalah ibu kota indonesia`

| Model | Perplexity Kalimat | Top-5 Prediksi Kata Berikutnya |
|---|---:|---|
| N=2 KN | 124,34 | yang (0,0463), dan (0,0242), pada (0,0163), di (0,0127), daftar (0,0125) |
| N=3 KN | 126,50 | daftar (0,1252), yang (0,0313), adalah (0,0304), di (0,0296), dengan (0,0277) |

Pada N=2, konteks yang dilihat adalah kata terakhir (`indonesia`), sedangkan pada N=3 konteks yang dilihat adalah dua kata terakhir (`kota indonesia`).

Contoh keluaran program demo untuk model N=2 KN:

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

> Catatan: program `04_inference_demo.py` sudah dikonfigurasi memakai `N_DEMO = 2` (model terbaik, N=2 KN), sehingga contoh keluaran di atas dapat direproduksi langsung.

---

## Analisis

### Pengaruh Smoothing

#### Kneser-Ney (KN) — Metode Terbaik dan Paling Efektif

- **Pengaruh:** Kneser-Ney secara konsisten menghasilkan nilai perplexity terendah di seluruh orde N.
- **Mengapa sangat bagus:** Kneser-Ney tidak sekadar menambah angka sembarangan pada frekuensi kata, melainkan memperhitungkan seberapa sering suatu kata bertindak sebagai kata penyambung (*continuation probability*) dan menerapkan *absolute discounting*. Pendekatan ini terbukti paling cerdas menangani n-gram yang tidak pernah muncul (*unseen n-grams*).

#### Add-k (k = 0,01) — Metode Menengah

- **Pengaruh:** Add-k menghasilkan perplexity yang jauh lebih rendah dibanding Laplace, tetapi masih kalah presisi dibanding Kneser-Ney.
- **Mengapa lebih baik dari Laplace:** Dengan menggunakan konstanta pecahan kecil (`k=0,01`), massa probabilitas dari kata-kata yang sering muncul tidak terpotong secara ekstrem untuk diberikan kepada kata langka.

#### Laplace (Add-1) — Metode Paling Buruk

- **Pengaruh:** Laplace smoothing menghasilkan nilai perplexity paling membengkak di semua nilai N.
- **Mengapa sangat buruk:** Penambahan `+1` pada pembagi yang melibatkan seluruh ukuran vocabulary (`V`) membuat model terlalu banyak mengambil massa probabilitas dari kata-kata yang wajar, lalu membagikannya secara tidak proporsional kepada kata-kata yang tidak pernah muncul. Fenomena ini disebut *over-smoothing*.

### Mengapa Orde Tinggi Justru Lebih Buruk?

Penyebab utamanya adalah **sparsitas data yang ekstrem**. Data training hanya sekitar 8,45 juta token, tetapi kosakata mencapai 472 ribu tipe. Untuk N=4:

- **94,53%** n-gram hanya muncul sekali.
- **89,60%** konteks hanya muncul sekali.
- Rata-rata satu konteks hanya diikuti **1,22** kata.

Artinya, untuk sebagian besar konteks 3 kata, model tidak memiliki cukup bukti statistik untuk memperkirakan distribusi kata berikutnya. Membandingkan N=2 dengan N=3/N=4 di sini bukan membandingkan “konteks lebih kaya vs lebih miskin”, melainkan “konteks yang cukup teramati vs konteks yang hampir selalu baru”.

Selain sparsitas, implementasi Kneser-Ney pada eksperimen ini melakukan **backoff langsung ke distribusi unigram kontinuasi**, bukan ke distribusi bigram. Akibatnya, ketika konteks trigram atau 4-gram tidak teramati, model kehilangan informasi kata sebelumnya yang sebenarnya sudah dimiliki oleh model bigram. Inilah sebabnya N=2 KN dapat mengungguli N=3 dan N=4 pada korpus ini.

### Bias Domain dan Kualitas Prediksi

Korpus Wikipedia memiliki distribusi topik yang khas. Banyak artikel tentang spesies, wilayah administratif, dan entitas langka. Dampaknya terlihat pada prediksi:

- `kucing` → `persia` masuk akal karena banyak artikel spesies/ras hewan.
- `ibu kota` → `kabupaten`, `kecamatan`, `provinsi` mencerminkan dominasi artikel wilayah administratif.
- `presiden` → `republik`, `amerika`, `ri` cukup wajar, tetapi N=3 memberi `juga` dan `dipilih` yang lebih lemah secara semantik.

Model n-gram memang tidak memahami makna; ia hanya menangkap pola lokal. Prediksi yang tampak “benar” sebagian berasal dari hafalan frasa di korpus, bukan pemahaman. Karena itu, evaluasi kualitatif harus dibaca sebagai ilustrasi, bukan bukti kualitas semantik.

---

## Kesimpulan

1. **Sembilan model n-gram berhasil dilatih dan disimpan** dari korpus Wikipedia bahasa Indonesia dengan kombinasi orde `N=2,3,4` dan smoothing Laplace, Add-k, serta Kneser-Ney.

2. **Model terbaik adalah N=2 dengan Kneser-Ney**, dengan **PPL 1.042,72** pada 500 baris test. Kneser-Ney juga merupakan smoothing terbaik pada semua orde, mengalahkan Add-k dan Laplace.

3. **Orde yang lebih tinggi tidak otomatis lebih baik pada implementasi ini.** Penyebab utamanya adalah sparsitas ekstrem (77–95% n-gram hapax; 60–90% konteks hanya muncul sekali) dan backoff KN yang langsung menuju unigram kontinuasi alih-alih bigram.

4. **Kualitas kualitatif juga tidak konsisten naik seiring orde.** N=3 unggul untuk `kucing → persia`, tetapi lebih lemah untuk `presiden`, `warna hijau`, dan PPL kalimat. N=2 KN lebih seimbang dan lebih murah secara komputasi.

5. **Bias domain korpus cukup kuat.** Prediksi seperti `ibu kota → kabupaten/kecamatan` mencerminkan distribusi artikel Wikipedia, bukan semantik bahasa Indonesia secara umum. Model n-gram hanya menangkap pola lokal.

6. **PPL masih tinggi** sehingga model ini belum layak disebut model bahasa siap produksi. Namun sebagai eksperimen komparatif, hasilnya valid dan memberi kesimpulan yang jelas: pada data dan implementasi ini, **bigram KN adalah konfigurasi terbaik**.
