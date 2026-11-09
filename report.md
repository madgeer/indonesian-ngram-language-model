# Laporan Eksperimen Language Model N-Gram pada Korpus Wikipedia Bahasa Indonesia

**Lokasi artefak:** `D:\Kuliah\semester 5\nlp\n-gram`
**Dataset:** `idwiki-latest-pages-articles.xml` (dump Wikipedia bahasa Indonesia)
**Model:** 9 model hasil training (`N = 2, 3, 4` × `Laplace, Add-k, Kneser-Ney`)
**Tanggal analisis:** 6 November 2026

---

## Ringkasan Eksekutif

Eksperimen berhasil membangun 9 model bahasa n-gram dari korpus Wikipedia bahasa Indonesia dan mengevaluasinya dengan perplexity (PPL). Hasil utama:

- **Model terbaik: N=2 dengan smoothing Kneser-Ney (KN), PPL 1.042,72** pada 500 baris uji.
- **Kneser-Ney konsisten menjadi smoothing terbaik** pada setiap orde n, jauh di atas Add-k dan Laplace.
- **Orde yang lebih tinggi tidak selalu lebih baik** pada implementasi ini: N=3 (PPL 1.622,97) dan N=4 (PPL 2.826,02) justru lebih buruk daripada N=2.
- Penyebab utamanya adalah **sparsitas data yang ekstrem** (77–95% n-gram hanya muncul sekali) dan **backoff KN yang disederhanakan**—model tidak benar-benar kembali ke distribusi bigram, melainkan langsung ke distribusi unigram kontinuasi.
- PPL absolut masih tinggi (±1.000), sehingga model ini lebih tepat diposisikan sebagai **eksperimen komparatif**, bukan model siap produksi.

---

## 1. Deskripsi

### 1.1 Tujuan Eksperimen

1. Membangun model bahasa n-gram dari korpus Wikipedia bahasa Indonesia.
2. Membandingkan pengaruh **orde n** (bigram, trigram, 4-gram) terhadap kualitas model.
3. Membandingkan tiga teknik **smoothing**: Laplace, Add-k, dan Kneser-Ney.
4. Mengevaluasi model secara kuantitatif (perplexity) dan kualitatif (prediksi kata berikutnya).
5. Menganalisis trade-off antara kualitas, ukuran model, dan latensi.

### 1.2 Dataset

Korpus berasal dari dump `idwiki-latest-pages-articles.xml` berukuran sekitar **5,93 GB**. Pemrosesan dilakukan dengan sampling **5%** artikel (parameter `sample_rate=0.05` pada pemanggilan `__main__`, meskipun nilai default fungsi adalah `0,20`).

| Properti | Data Train | Data Test |
|---|---:|---:|
| Jumlah artikel | 31.520 | 7.881 |
| Jumlah token | 8.451.690 | 2.075.431 |
| Jumlah tipe kata (vocabulary) | 475.762 | 170.030 |
| Ukuran file | 61,6 MB | 15,1 MB |
| Proporsi | 80% | 20% |

Karakteristik pemrosesan:

- Hanya mengambil halaman dengan **namespace 0** (artikel utama).
- Mengabaikan halaman **redirect** (`#ALIH`, `#REDIRECT`).
- Membersihkan wikitext: komentar, `<ref>`, template `{{...}}`, pranala kategori/berkas, heading, URL, dan karakter non-alfanumerik.
- Hanya menyimpan artikel dengan panjang teks bersih **> 50 karakter**.
- Mengacak artikel dengan `random.seed(42)`, lalu membagi **80% train / 20% test**.
- Satu baris file korpus merepresentasikan satu artikel.

**Catatan OOV:** sekitar **43,47% tipe kata** dan **5,15% token** pada data test tidak terdapat di kosakata train. Model tidak memiliki token `<UNK>`, sehingga kata OOV hanya mendapat probabilitas sangat kecil dari rumus smoothing.

### 1.3 Program yang Digunakan

| File | Peran | Status |
|---|---|---|
| `01-clean-corpus.py` | Parsing XML, pembersihan wikitext, sampling, split train/test | Sintaks valid; membutuhkan XML 5,93 GB |
| `02_run_experiments.py` | Definisi kelas model, training, penyimpanan `.pkl` | Berjalan |
| `03_generate_report_table.py` | Memuat model, menghitung PPL, mencetak tabel | Berjalan (impor diperbaiki dengan `importlib`) |
| `04_inference_demo.py` | Demo inferensi interaktif dan prediksi top-5 | Berjalan (impor diperbaiki dengan `importlib`) |

#### `01-clean-corpus.py`

- `clean_wikitext()` menghapus komentar HTML, referensi, template, pranala wiki (label dipertahankan), heading, URL, dan karakter selain `a-z`, `A-Z`, `0-9`, spasi, serta tanda baca `. , ! ?`.
- `process_and_split_corpus()` memakai `ElementTree.iterparse` agar XML besar tidak dimuat sekaligus ke memori.
- Hasil akhir ditulis ke `train_corpus.txt` dan `test_corpus.txt`.

#### `02_run_experiments.py`

Kelas `ComprehensiveNGram` menyimpan:

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

Alur evaluasi `evaluate_ppl()`:

- Mengambil **500 baris pertama** data test yang panjangnya **> 30 karakter**.
- Menghitung `log2` probabilitas setiap token target, termasuk `<EOS>`.
- PPL dihitung dengan basis 2: `2^(-rata-rata log2 P)`.

#### `03_generate_report_table.py`

Memuat setiap `.pkl`, menjalankan `evaluate_ppl()`, lalu mencetak tabel PPL. Awalnya file ini gagal karena baris:

```python
from 02_run_experiments import ComprehensiveNGram, TEST_FILE
```

Nama modul yang diawali angka bukan identifier Python yang valid, sehingga menimbulkan `SyntaxError`. Perbaikan yang diterapkan adalah memuat modul melalui `importlib`:

```python
_core = importlib.import_module("02_run_experiments")
ComprehensiveNGram = _core.ComprehensiveNGram
sys.modules["__main__"].ComprehensiveNGram = ComprehensiveNGram
```

Baris terakhir diperlukan karena `.pkl` menyimpan referensi ke `__main__.ComprehensiveNGram`, bukan ke `02_run_experiments.ComprehensiveNGram`.

#### `04_inference_demo.py`

Menyediakan demo interaktif:

- Menghitung PPL satu kalimat.
- Menampilkan top-5 prediksi kata berikutnya.
- Menampilkan latensi komputasi.

Semula juga gagal dijalankan karena:

```python
from 01_run_experiments import ComprehensiveNGram
```

Selain `SyntaxError` akibat nama modul diawali angka, nama file yang diimpor juga salah (seharusnya `02_run_experiments.py`, bukan `01_run_experiments.py`). Perbaikan yang diterapkan: modul `02_run_experiments` dimuat dengan `importlib`, lalu kelasnya didaftarkan ke `__main__` agar `.pkl` dapat di-unpickle. Ditambahkan pula `sys.stdout.reconfigure(encoding="utf-8")` agar karakter `✓` tidak memicu `UnicodeEncodeError` di konsol Windows.

Demo saat ini masih memakai `N=3, KN`. Hasil eksperimen menunjukkan model terbaik adalah `N=2, KN`, sehingga `N_DEMO` dapat diubah menjadi `2` bila ingin demo memakai model terbaik.

### 1.4 Konfigurasi Model

| Parameter | Nilai |
|---|---|
| Orde n | 2, 3, 4 |
| Smoothing | Laplace, Add-k, Kneser-Ney |
| Nilai k (Add-k) | 0,01 |
| Nilai d (KN) | 0,75 |
| Tokenisasi | `\b\w+\b`, lowercase |
| Pemecah kalimat | `[.!?]+` |
| Padding | `<BOS>` sebanyak `n-1`, diakhiri `<EOS>` |
| Evaluasi PPL | 500 baris test, panjang > 30 karakter |
| Basis log | 2 |
| Seed pengacakan korpus | 42 |
| Python | 3.13.1 |

Rumus probabilitas yang diimplementasikan:

```text
Laplace : P(w|h) = (c(h,w) + 1) / (c(h) + V)

Add-k   : P(w|h) = (c(h,w) + k) / (c(h) + k·V)

KN      : P(w|h) = max(c(h,w) - d, 0) / c(h) + λ(h)·P_cont(w)
          λ(h)   = (d / c(h)) · |{w' : c(h,w') > 0}|
          P_cont(w) = N1+(*w) / N1+(* *)
```

Keterangan:

- `c(h,w)` = frekuensi n-gram konteks `h` diikuti kata `w`.
- `c(h)` = frekuensi konteks `h`.
- `V` = ukuran kosakata train.
- `N1+(*w)` = jumlah kata berbeda yang pernah mendahului `w`.
- `N1+(* *)` = jumlah total pasangan kata berbeda.

Jika `c(h) = 0`, implementasi KN langsung mengembalikan `P_cont(w)`. Jika `P_cont(w) = 0` (kata OOV), digunakan fallback `1/V`.

> **Penting:** Implementasi KN di sini adalah **interpolated absolute discounting yang disederhanakan**, bukan KN rekursif penuh. Backoff dari trigram/4-gram tidak menuju distribusi bigram, melainkan langsung ke distribusi unigram kontinuasi. Detail ini akan menjadi kunci interpretasi hasil.

---

## 2. Eksperimen

### 2.1 Prosedur

1. Korpus dibersihkan dan dibagi menjadi train/test 80:20.
2. Sembilan model dilatih dengan kombinasi `n ∈ {2,3,4}` dan `smoothing ∈ {laplace, add_k, kn}`.
3. Setiap model disimpan sebagai `ngram_model_N{n}_{smoothing}.pkl`.
4. Setiap model dievaluasi dengan `evaluate_ppl()` pada 500 baris test yang sama.
5. Model terbaik diuji secara kualitatif pada beberapa konteks bahasa Indonesia.

Program `02_run_experiments.py` memiliki mekanisme cache: jika file `.pkl` sudah ada, training dilewati. Waktu training juga hanya dicetak ke layar dan **tidak disimpan** sebagai artefak.

### 2.2 Hasil Perplexity

PPL lebih rendah lebih baik. Angka di bawah dihitung ulang dari 9 file `.pkl` menggunakan metode evaluasi yang sama dengan `03_generate_report_table.py`.

| Orde N | Laplace | Add-k | Kneser-Ney | Model Terbaik |
|---:|---:|---:|---:|---|
| N=2 | 40.036,65 | 4.947,80 | **1.042,72** | **KN** |
| N=3 | 158.628,63 | 43.225,03 | **1.622,97** | **KN** |
| N=4 | 229.838,58 | 111.013,48 | **2.826,02** | **KN** |

Urutan performa:

```text
N=2 KN  (1.042,72)  <  N=3 KN  (1.622,97)  <  N=4 KN  (2.826,02)
```

Pada semua orde, urutan smoothing konsisten:

```text
KN  <  Add-k  <  Laplace
```

### 2.3 Validasi pada Subset Token yang Sama

Perbandingan PPL di atas memiliki satu potensi bias: jumlah token yang dievaluasi berbeda antar orde karena kalimat pendek dilewati. Untuk mengujinya, dilakukan evaluasi tambahan pada **6.898 kalimat dan 135.079 target token yang identik** untuk semua model.

| Orde N | Laplace | Add-k | Kneser-Ney |
|---:|---:|---:|---:|
| N=2 | 46.061,84 | 5.191,18 | **1.022,54** |
| N=3 | 190.417,59 | 49.686,62 | **1.592,38** |
| N=4 | 271.273,61 | 126.921,84 | **2.714,87** |

Hasilnya tetap sama: **N=2 KN terbaik**, diikuti N=3 dan N=4. Jadi penurunan performa pada orde tinggi **bukan** sekadar artefak perbedaan jumlah token atau ukuran kosakata.

### 2.4 Karakteristik Model dan Sparsitas

| Orde | Kosakata | N-gram unik | Konteks unik | N-gram hapax | Konteks muncul 1× | Rata-rata lanjutan per konteks |
|---:|---:|---:|---:|---:|---:|---:|
| N=2 | 472.653 | 3.365.725 | 472.654 | 77,07% | 59,82% | 7,12 |
| N=3 | 468.044 | 6.030.124 | 3.269.420 | 89,27% | 77,42% | 1,84 |
| N=4 | 461.958 | 6.999.302 | 5.742.154 | 94,53% | 89,60% | 1,22 |

Kosakata menurun seiring bertambahnya n karena kalimat dengan panjang `< n` dilewati saat training. Perbedaan ini kecil, tetapi tetap memengaruhi denominator smoothing.

Temuan penting:

- Pada N=2, satu konteks rata-rata diikuti **7,12 kata berbeda**.
- Pada N=3, rata-rata turun menjadi **1,84**.
- Pada N=4, rata-rata hanya **1,22**; artinya hampir setiap konteks 3 kata hanya pernah diikuti satu kata.
- **94,53%** n-gram N=4 hanya muncul satu kali.

### 2.5 Uji Kualitatif: Prediksi Kata Berikutnya

Diambil dari model KN terbaik pada masing-masing orde. Untuk konteks multi-kata, N=2 hanya melihat kata terakhir, sedangkan N=3 melihat dua kata terakhir. Untuk input satu kata, N=3 menggunakan konteks `<BOS> + kata`, sedangkan N=2 hanya menggunakan kata tersebut.

| Konteks | N=2 KN (top-5) | N=3 KN (top-5) |
|---|---|---|
| `kucing` | yang (0,1015), persia (0,0765), dan (0,0458), ini (0,0424), besar (0,0293) | persia (0,3084), ini (0,1757), yang (0,0120), merupakan (0,0090), menjadi (0,0089) |
| `presiden` | republik (0,0364), dan (0,0320), amerika (0,0290), ri (0,0258), indonesia (0,0227) | juga (0,0299), dipilih (0,0289), indonesia (0,0237), amerika (0,0236), republik (0,0235) |
| `ibu kota` | ini (0,0682), di (0,0356), yang (0,0319), dan (0,0271), depok (0,0173) | kabupaten (0,1229), kecamatan (0,0791), provinsi (0,0663), negara (0,0620), dari (0,0320) |
| `pada tahun` | yang (0,0203), 2010 (0,0151), 2016 (0,0129), 2012 (0,0127), 2007 (0,0126) | 2010 (0,0184), 2009 (0,0169), yang (0,0168), 2006 (0,0167), 2007 (0,0164) |
| `warna hijau` | dan (0,1058), yang (0,0621), tua (0,0292), di (0,0287), dengan (0,0203) | dan (0,0751), pada (0,0707), yang (0,0429), adalah (0,0422), atau (0,0396) |

Interpretasi singkat:

- N=3 lebih spesifik untuk `kucing` → `persia` (0,3084), jauh lebih tinggi daripada N=2 (0,0765).
- Namun N=3 justru lebih buruk untuk `presiden` → `juga/dipilih`, sementara N=2 memberi `republik/amerika/indonesia` yang lebih masuk akal.
- `ibu kota` → `kabupaten/kecamatan/provinsi` mencerminkan bias domain korpus Wikipedia yang banyak memuat artikel wilayah administratif, bukan semata-mata kesalahan model.
- N=2 `warna hijau` → `tua` lebih baik daripada N=3 yang memberi `dan/pada`.

### 2.6 PPL per Kalimat

| Kalimat | N=2 KN | N=3 KN |
|---|---:|---:|
| `jakarta adalah ibu kota indonesia` | 124,34 | 126,50 |
| `presiden indonesia pada tahun 2020 adalah joko widodo` | **117,41** | 421,55 |
| `kucing adalah hewan mamalia yang lucu` | **294,27** | 824,54 |
| `warna hijau biru langit sangat indah sekali` | **309,23** | 1.727,92 |

N=2 KN lebih rendah pada 3 dari 4 kalimat. Ini konsisten dengan hasil agregat: pada korpus ini, bigram KN memberikan generalisasi yang lebih baik daripada trigram KN yang rapuh terhadap konteks yang jarang.

### 2.7 Latensi dan Ukuran Model

| Orde | Ukuran `.pkl` (Laplace/Add-k) | Ukuran `.pkl` (KN) | Waktu load (KN) |
|---:|---:|---:|---:|
| N=2 | 74,14 MiB | 115,41 MiB | 6,47 detik |
| N=3 | 204,07 MiB | 245,07 MiB | 11,35 detik |
| N=4 | 337,28 MiB | 377,94 MiB | 15,43 detik |

Total seluruh artefak model sekitar **1,92 GiB**.

Latensi inferensi yang terukur:

- Prediksi top-5: **238–299 ms** untuk N=2 dan **487–582 ms** untuk N=3.
- PPL satu kalimat: **0,03–0,08 ms**.
- Evaluasi 500 baris test: **0,30–0,52 detik** per model.

Latensi prediksi top-5 relatif tinggi karena `predict_next_word()` memindai **seluruh** `ngram_counts` untuk mencari kandidat dengan konteks yang cocok. Kompleksitas ini akan memburuk secara linear terhadap jumlah n-gram.

---

## 3. Analisis

### 3.1 Pengaruh Smoothing

Kneser-Ney unggul sangat konsisten:

| Perbandingan | N=2 | N=3 | N=4 |
|---|---:|---:|---:|
| PPL Laplace ÷ PPL KN | 38,4× | 97,7× | 81,3× |
| PPL Add-k ÷ PPL KN | 4,75× | 26,6× | 39,3× |
| PPL Laplace ÷ PPL Add-k | 8,09× | 3,67× | 2,07× |

**Laplace paling buruk** karena ukuran kosakata sangat besar (462–473 ribu). Dengan rumus `(c+1)/(c+V)`, setiap kata mendapat tambahan massa 1, sehingga probabilitas kata yang benar-benar muncul menjadi sangat kecil. Sebagai ilustrasi, pada N=3 (`V=468.044`) dan konteks dengan `c(h)=1`, Laplace memberi sekitar `2/(1+468.044)`, yaitu hanya sekitar `4,3 × 10⁻⁶`.

**Add-k** memperbaiki Laplace karena denominator menjadi `c(h) + k·V`. Dengan `k=0,01`, `k·V` hanya sekitar 4.600–4.700, bukan 468.000. Akibatnya probabilitas kata yang teramati naik cukup drastis dan PPL membaik 2–8 kali lipat dibanding Laplace. Namun Add-k masih membagi massa ke seluruh kosakata secara seragam.

**Kneser-Ney** jauh lebih baik karena menerapkan **absolute discounting** dan mengalokasikan massa diskon ke distribusi kontinuasi, bukan menyebarkannya ke semua kata secara rata. Pada N=2, KN 4,75× lebih baik daripada Add-k; pada N=4, KN 39,3× lebih baik. Semakin tinggi orde, semakin besar keuntungan KN karena semakin banyak konteks yang harus di-backoff.

### 3.2 Mengapa Orde Tinggi Justru Lebih Buruk?

Secara teori, menambah konteks seharusnya menurunkan perplexity. Namun pada eksperimen ini terjadi sebaliknya:

```text
N=2 KN: 1.042,72
N=3 KN: 1.622,97  (+55,6% dari N=2)
N=4 KN: 2.826,02  (+74,1% dari N=3; 2,71× N=2)
```

Penyebab utamanya ada tiga.

#### a. Sparsitas ekstrem

Data training hanya sekitar 8,45 juta token, tetapi kosakata mencapai 472 ribu tipe. Untuk N=4:

- 94,53% n-gram hanya muncul sekali.
- 89,60% konteks hanya muncul sekali.
- Rata-rata satu konteks hanya diikuti 1,22 kata.

Artinya, untuk sebagian besar konteks 3 kata, model tidak memiliki cukup bukti statistik untuk memperkirakan distribusi kata berikutnya. Membandingkan N=2 dengan N=3/N=4 di sini bukan membandingkan “konteks lebih kaya vs lebih miskin”, melainkan “konteks yang cukup teramati vs konteks yang hampir selalu baru”.

#### b. Backoff KN tidak rekursif

Ini faktor yang paling penting. Pada KN yang benar, trigram yang tidak teramati harus di-backoff ke **distribusi bigram**:

```text
P(w | a,b) → P(w | b) → P(w)
```

Implementasi saat ini melakukan:

```text
P(w | a,b) → P_cont(w)   (distribusi unigram kontinuasi)
```

Ketika `c(a,b) = 0`, model langsung mengembalikan `P_cont(w)` dan **mengabaikan kata `b`** yang sebenarnya informatif. Akibatnya, untuk N=3 dan N=4, mayoritas prediksi kehilangan konteks terdekatnya. Inilah alasan utama mengapa N=3 dan N=4 lebih buruk daripada N=2: mereka memang menggunakan konteks yang lebih panjang ketika konteks itu teramati, tetapi ketika konteks tidak teramati—dan itu mayoritas—mereka membuang konteks yang sudah dimiliki N=2.

Selain itu, untuk konteks yang hanya muncul sekali (`c(h)=1`, `unique_follows=1`), KN memberi bobot:

```text
0,25 pada kata yang teramati + 0,75 · P_cont(w)
```

Tiga perempat massa langsung dialihkan ke distribusi unigram yang tidak mempertimbangkan kata sebelumnya. Pada KN penuh, massa tersebut akan dialihkan ke bigram yang jauh lebih tajam.

#### c. Penalti OOV dan kosakata tertutup

Tidak ada token `<UNK>`. Kata OOV hanya mendapat:

- Laplace: sekitar `1/(c(h)+V)`
- Add-k: sekitar `k/(c(h)+k·V)`
- KN: `λ(h) · (1/V)` jika `P_cont(w)=0`

Untuk konteks yang sering muncul, `λ(h)` bisa sangat kecil, sehingga kata OOV mendapat probabilitas yang sangat rendah. Besarnya penalti bergantung pada distribusi konteks dan nilai `λ(h)`, sehingga tidak otomatis lebih besar pada orde yang lebih tinggi.

Untuk memisahkan efek OOV, evaluasi diulang dengan **mengeluarkan token OOV** dari perhitungan PPL:

| Orde | PPL semua token | PPL hanya token in-vocabulary |
|---:|---:|---:|
| N=2 KN | 1.042,72 | 946,69 |
| N=3 KN | 1.622,97 | 1.517,14 |
| N=4 KN | 2.826,02 | 2.691,08 |

Kesimpulannya, OOV memang menaikkan PPL (sekitar 5–10%), tetapi **bukan penyebab utama** keterpurukan N=3 dan N=4. Penyebab utamanya tetap sparsitas dan backoff yang tidak rekursif.

### 3.3 Bias Domain dan Kualitas Prediksi

Korpus Wikipedia memiliki distribusi topik yang khas. Banyak artikel tentang spesies, wilayah administratif, dan entitas langka. Dampaknya terlihat pada prediksi:

- `kucing` → `persia` masuk akal karena banyak artikel spesies/ras hewan.
- `ibu kota` → `kabupaten`, `kecamatan`, `provinsi` mencerminkan dominasi artikel wilayah administratif.
- `presiden` → `republik`, `amerika`, `ri` cukup wajar, tetapi N=3 memberi `juga` dan `dipilih` yang lebih lemah secara semantik.

Model n-gram memang tidak memahami makna; ia hanya menangkap pola lokal. Prediksi yang tampak “benar” sebagian berasal dari hafalan frasa di korpus, bukan pemahaman. Karena itu, evaluasi kualitatif harus dibaca sebagai ilustrasi, bukan bukti kualitas semantik.

### 3.4 Trade-off Kualitas vs Biaya

| Aspek | N=2 KN | N=3 KN | N=4 KN |
|---|---:|---:|---:|
| PPL (500 baris) | **1.042,72** | 1.622,97 | 2.826,02 |
| Ukuran model | 115,41 MiB | 245,07 MiB | 377,94 MiB |
| Waktu load | 6,47 s | 11,35 s | 15,43 s |
| Latensi prediksi top-5 | 238–299 ms | 487–582 ms | lebih tinggi (pemindaian lebih besar) |
| Kebutuhan memori | rendah | sedang | tinggi |

N=2 KN adalah pilihan paling masuk akal secara biaya-manfaat: kualitas terbaik, ukuran paling kecil, dan latensi paling rendah. N=3/N=4 tidak memberikan keuntungan kualitas yang sebanding dengan kenaikan ukuran dan latensi.

### 3.5 Validitas dan Keterbatasan

1. **Evaluasi hanya 500 baris pertama**, bukan seluruh 7.881 artikel test. Baris diambil secara berurutan, bukan acak, sehingga bisa mengandung bias topik.
2. **`max_sentences=500` sebenarnya membatasi jumlah baris/artikel**, bukan jumlah kalimat. Setiap baris dapat berisi banyak kalimat.
3. **Ukuran kosakata berbeda antar orde** karena kalimat pendek dilewati. Evaluasi pada subset token yang sama sudah dilakukan dan tidak mengubah kesimpulan, tetapi angka PPL resmi tetap dipengaruhi perbedaan ini.
4. **Tidak ada validation set** dan tidak ada tuning hyperparameter. Nilai `k=0,01` dan `d=0,75` ditetapkan tetap, bukan hasil pencarian.
5. **Tidak ada token `<UNK>`** dan tidak ada strategi penanganan OOV yang eksplisit.
6. **Tidak ada pencatatan metrik terstruktur.** Waktu training, konfigurasi, dan hasil PPL tidak disimpan otomatis ke JSON/CSV, sehingga reproduksibilitas bergantung pada file `.pkl`.
7. **Skrip 03 dan 04 sempat tidak dapat dijalankan** karena `SyntaxError` pada nama modul yang diawali angka. Keduanya sudah diperbaiki dengan `importlib`. Namun `04_inference_demo.py` masih meng-hardcode N=3 KN, padahal hasil menunjukkan N=2 KN adalah model terbaik.
8. **PPL hanya metrik intrinsik.** Tidak ada evaluasi downstream (klasifikasi, terjemahan, generasi, dsb.), sehingga belum diketahui apakah perbedaan PPL berdampak pada tugas nyata.
9. **PPL absolut tinggi** (±1.000) untuk ukuran model bahasa praktis. Ini wajar untuk korpus 5% dengan kosakata 472 ribu dan tanpa neural smoothing, tetapi perlu ditekankan agar hasil tidak disalahartikan sebagai model siap pakai.

### 3.6 Catatan Reproduksibilitas

Pada audit awal, `03_generate_report_table.py` dan `04_inference_demo.py` belum dapat dijalankan akibat baris impor yang tidak valid. Angka pada laporan ini dihitung ulang dengan memuat `02_run_experiments.py` melalui `importlib` dan menyuntikkan kelas `ComprehensiveNGram` ke `__main__` agar `.pkl` dapat di-unpickle. Metode evaluasi yang dipanggil tetap `model.evaluate_ppl(TEST_FILE, max_sentences=500)`, sama seperti desain awal.

Setelah perbaikan, `03_generate_report_table.py` berhasil dijalankan dan menghasilkan angka yang sama dengan tabel di laporan ini. Pola impor yang sekarang dipakai:

```python
import importlib
import sys

_core = importlib.import_module("02_run_experiments")
ComprehensiveNGram = _core.ComprehensiveNGram
sys.modules["__main__"].ComprehensiveNGram = ComprehensiveNGram
```

Langkah lanjutan yang disarankan adalah mengubah `N_DEMO` pada `04_inference_demo.py` menjadi `2` agar demo memakai model terbaik (`N=2, KN`).

---

## 4. Kesimpulan

1. **Sembilan model n-gram berhasil dilatih dan disimpan** dari korpus Wikipedia bahasa Indonesia dengan kombinasi orde `N=2,3,4` dan smoothing `Laplace, Add-k, Kneser-Ney`.

2. **Model terbaik adalah N=2 dengan Kneser-Ney**, dengan **PPL 1.042,72** pada 500 baris test. Kneser-Ney juga merupakan smoothing terbaik pada semua orde, mengalahkan Add-k hingga 39× dan Laplace hingga 98×.

3. **Orde yang lebih tinggi tidak otomatis lebih baik pada implementasi ini.** PPL meningkat dari N=2 (1.042,72) ke N=3 (1.622,97) dan N=4 (2.826,02). Penyebab utamanya adalah sparsitas ekstrem (77–95% n-gram hapax; 60–90% konteks hanya muncul sekali) dan backoff KN yang langsung menuju unigram kontinuasi alih-alih bigram.

4. **Kualitas kualitatif juga tidak konsisten naik seiring orde.** N=3 unggul untuk `kucing → persia`, tetapi lebih lemah untuk `presiden`, `warna hijau`, dan PPL kalimat. N=2 KN lebih seimbang dan lebih murah secara komputasi.

5. **Bias domain korpus cukup kuat.** Prediksi seperti `ibu kota → kabupaten/kecamatan` mencerminkan distribusi artikel Wikipedia, bukan semantik bahasa Indonesia secara umum. Model n-gram hanya menangkap pola lokal.

6. **PPL absolut masih tinggi** sehingga model ini belum layak disebut model bahasa siap produksi. Namun sebagai eksperimen komparatif, hasilnya valid dan memberi kesimpulan yang jelas: pada data dan implementasi ini, **bigram KN adalah konfigurasi terbaik**.

### Rekomendasi

1. **Pertahankan impor berbasis `importlib`** pada `03` dan `04` (sudah diterapkan), lalu ubah `N_DEMO` pada `04` menjadi `2` agar demo memakai model terbaik.
2. **Implementasikan KN rekursif penuh** dengan backoff trigram → bigram → unigram, lalu bandingkan ulang N=2 vs N=3 vs N=4. Dengan backoff yang benar, trigram berpotensi mengungguli bigram.
3. **Tambahkan token `<UNK>` dan vocabulary pruning** (misalnya kata dengan frekuensi ≥ 2) untuk mengurangi `V`, memperbaiki smoothing, dan menekan ukuran model.
4. **Gunakan validation set** untuk mencari `k` dan `d` secara empiris, bukan menetapkan `0,01` dan `0,75` secara tetap.
5. **Evaluasi pada seluruh test set** dan beberapa subset acak, lalu simpan hasil ke JSON/CSV agar reproducible.
6. **Optimalkan prediksi** dengan indeks `context → daftar kata berikutnya`, sehingga tidak perlu memindai seluruh `ngram_counts`.
7. **Untuk peningkatan kualitas yang signifikan**, pertimbangkan KenLM, model neural (LSTM/Transformer), atau setidaknya corpus yang lebih besar; untuk kebutuhan latensi rendah dan sumber daya terbatas, **N=2 KN** sudah menjadi baseline yang kuat.

---

## Lampiran A. Inventaris Model

| File | Orde | Smoothing | PPL | Ukuran (MiB) | Load (detik) |
|---|---:|---|---:|---:|---:|
| `ngram_model_N2_laplace.pkl` | 2 | Laplace | 40.036,65 | 74,14 | 2,88 |
| `ngram_model_N2_add_k.pkl` | 2 | Add-k | 4.947,80 | 74,14 | 3,04 |
| `ngram_model_N2_kn.pkl` | 2 | Kneser-Ney | **1.042,72** | 115,41 | 6,47 |
| `ngram_model_N3_laplace.pkl` | 3 | Laplace | 158.628,63 | 204,07 | 8,24 |
| `ngram_model_N3_add_k.pkl` | 3 | Add-k | 43.225,03 | 204,07 | 8,37 |
| `ngram_model_N3_kn.pkl` | 3 | Kneser-Ney | 1.622,97 | 245,07 | 11,35 |
| `ngram_model_N4_laplace.pkl` | 4 | Laplace | 229.838,58 | 337,28 | 12,27 |
| `ngram_model_N4_add_k.pkl` | 4 | Add-k | 111.013,48 | 337,28 | 12,08 |
| `ngram_model_N4_kn.pkl` | 4 | Kneser-Ney | 2.826,02 | 377,94 | 15,43 |

## Lampiran B. Ringkasan Angka Kunci

| Metrik | Nilai |
|---|---:|
| Artikel train / test | 31.520 / 7.881 |
| Token train / test | 8.451.690 / 2.075.431 |
| Tipe kata train / test | 475.762 / 170.030 |
| OOV test (tipe / token) | 43,47% / 5,15% |
| Model terbaik | N=2, Kneser-Ney |
| PPL terbaik | 1.042,72 |
| PPL terburuk | 229.838,58 (N=4, Laplace) |
| Rentang ukuran model | 74,14–377,94 MiB |
| Total artefak model | ±1,92 GiB |
| N-gram hapax N=2 / N=3 / N=4 | 77,07% / 89,27% / 94,53% |
| Konteks muncul 1× N=2 / N=3 / N=4 | 59,82% / 77,42% / 89,60% |
