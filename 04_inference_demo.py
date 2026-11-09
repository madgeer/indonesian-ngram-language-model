import re
import math
import time
import os
import sys
import pickle
import importlib

# Pastikan karakter Unicode (✓) tidak memicu UnicodeEncodeError di Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Konfigurasi Model Terbaik yang Di-Load untuk Demo
N_DEMO = 2
SMOOTHING_DEMO = "kn"
MODEL_FILE = f"ngram_model_N{N_DEMO}_{SMOOTHING_DEMO}.pkl"

# Nama file diawali angka, sehingga tidak bisa ditulis sebagai:
# from 02_run_experiments import ...
# Gunakan importlib dengan nama modul berupa string.
_core = importlib.import_module("02_run_experiments")
ComprehensiveNGram = _core.ComprehensiveNGram

# Pickle menyimpan referensi ke __main__.ComprehensiveNGram saat model dilatih
# oleh 02_run_experiments.py. Pastikan nama tersebut tersedia saat unpickle.
sys.modules["__main__"].ComprehensiveNGram = ComprehensiveNGram

def predict_next_word(model, text, top_k=5):
    tokens = re.findall(r'\b\w+\b', text.lower())
    ctx_len = model.n - 1
    
    if len(tokens) < ctx_len:
        context = tuple(['<BOS>'] * (ctx_len - len(tokens)) + tokens)
    else:
        context = tuple(tokens[-ctx_len:])
        
    candidates = set()
    for ngram in model.ngram_counts.keys():
        if ngram[:-1] == context:
            candidates.add(ngram[-1])
            
    if not candidates:
        candidates = set(list(model.vocab)[:100])
        
    candidates.discard('<BOS>')
    candidates.discard('<EOS>')

    scores = []
    for w in candidates:
        prob = model.get_probability(w, context)
        scores.append((w, prob))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def calculate_sentence_ppl(model, sentence):
    tokens = model.tokenize(sentence)
    if len(tokens) < model.n:
        return float('inf')
    
    log_prob_sum = 0.0
    n_eval = len(tokens) - (model.n - 1)
    
    for i in range(model.n - 1, len(tokens)):
        context = tuple(tokens[i - (model.n - 1):i])
        word = tokens[i]
        prob = model.get_probability(word, context)
        log_prob_sum += math.log2(prob)
        
    return math.pow(2, -log_prob_sum / n_eval)

if __name__ == "__main__":
    print("="*65)
    print("  [3/3] INFERENCE N-GRAM")
    print("="*65)
    
    if not os.path.exists(MODEL_FILE):
        print(f"[!] Model '{MODEL_FILE}' belum ada. Jalankan 02_run_experiments.py dulu!")
        exit()

    t0 = time.time()
    print(f"[*] Memuat model '{MODEL_FILE}' ke memori...")
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print(f"[✓] Model Siap Digunakan dalam {time.time() - t0:.2f} detik!\n")

    print(f"Status Model Active: Orde N={model.n} | Smoothing={model.smoothing.upper()}")
    print("Ketik kalimat input untuk menguji Perplexity & Prediksi Kata.")
    print("Ketik 'q' untuk keluar.\n")

    while True:
        prompt = input("Masukkan kalimat: ").strip()
        if prompt.lower() == 'q':
            break
        if not prompt:
            continue
            
        t_inf = time.perf_counter()
        ppl_val = calculate_sentence_ppl(model, prompt)
        predictions = predict_next_word(model, prompt, top_k=5)
        latency = (time.perf_counter() - t_inf) * 1000

        print(f"\n[+] Hasil Perplexity Kalimat : {ppl_val:.2f}")
        print(f"[+] Latensi Komputasi        : {latency:.2f} ms")
        print(f"[+] Top-5 Prediksi Kata Berikutnya:")
        for idx, (word, prob) in enumerate(predictions, 1):
            print(f"    {idx}. {word:<18} (Prob: {prob:.4f})")
        print("-" * 50)