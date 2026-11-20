"""Hitung perplexity semua model dan cetak tabel hasil eksperimen."""

import os
import sys
import pickle
import importlib

# Pastikan karakter Unicode (✓, ↓) tidak memicu UnicodeEncodeError di Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Nama file diawali angka, sehingga tidak bisa ditulis sebagai:
# from 02_run_experiments import ...
# Gunakan importlib dengan nama modul berupa string.
_core = importlib.import_module("02_run_experiments")
ComprehensiveNGram = _core.ComprehensiveNGram
TEST_FILE = _core.TEST_FILE

# Pickle menyimpan referensi ke __main__.ComprehensiveNGram saat model dilatih
# oleh 02_run_experiments.py. Pastikan nama tersebut tersedia saat unpickle.
sys.modules["__main__"].ComprehensiveNGram = ComprehensiveNGram

if __name__ == "__main__":
    n_options = [2, 3, 4]
    smoothing_options = ["laplace", "add_k", "kn"]
    results = []

    print("="*60)
    print("  [2/3] EVALUASI PERPLEXITY GLOBAL & GENERATE TABEL LAPORAN")
    print("="*60)

    for n in n_options:
        for sm in smoothing_options:
            pkl_file = f"ngram_model_N{n}_{sm}.pkl"
            if not os.path.exists(pkl_file):
                print(f"[!] File {pkl_file} tidak ditemukan. Jalankan 02_run_experiments.py dulu!")
                continue

            with open(pkl_file, 'rb') as f:
                model = pickle.load(f)

            ppl = model.evaluate_ppl(TEST_FILE, max_sentences=500)
            results.append((n, sm.upper(), ppl))
            print(f"[✓] Evaluasi N={n} | {sm.upper():<7} -> PPL: {ppl:.2f}")

    print("\n" + "="*50)
    print("      TABEL HASIL EKSPERIMEN")
    print("="*50)
    print(f"{'Orde N':<8} | {'Smoothing':<12} | {'Perplexity (PPL) ↓':<20}")
    print("-" * 50)
    for n, sm, ppl in results:
        print(f"N={n:<6} | {sm:<12} | {ppl:<20.2f}")
    print("="*50)