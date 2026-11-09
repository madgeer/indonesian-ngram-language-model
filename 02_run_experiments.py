import re
from collections import Counter
import math
import time
import os
import pickle

TRAIN_FILE = "train_corpus.txt"
TEST_FILE = "test_corpus.txt"

class ComprehensiveNGram:
    def __init__(self, n=3, smoothing="kn", k=0.01, d=0.75):
        self.n = n
        self.smoothing = smoothing.lower()
        self.k = k
        self.d = d
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self.continuation_counts = Counter()
        self.seen_preceding_pairs = set()
        self.context_unique_follows = Counter()

    def tokenize(self, sentence):
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        if not tokens:
            return []
        return ['<BOS>'] * (self.n - 1) + tokens + ['<EOS>']

    def train(self, corpus_path):
        t0 = time.time()
        sent_split = re.compile(r'[.!?]+')
        word_token = re.compile(r'\b\w+\b')
        
        n = self.n
        is_kn = (self.smoothing == "kn")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                for sentence in sent_split.split(line):
                    raw_tokens = word_token.findall(sentence.lower())
                    if len(raw_tokens) < n:
                        continue
                    tokens = ['<BOS>'] * (n - 1) + raw_tokens + ['<EOS>']
                    self.vocab.update(raw_tokens)
                    
                    for i in range(len(tokens) - n + 1):
                        ngram = tuple(tokens[i:i + n])
                        context = ngram[:-1]
                        word = ngram[-1]
                        
                        if ngram not in self.ngram_counts:
                            self.context_unique_follows[context] += 1
                        
                        self.ngram_counts[ngram] += 1
                        self.context_counts[context] += 1
                        
                        if is_kn:
                            pair = (context[-1], word)
                            if pair not in self.seen_preceding_pairs:
                                self.seen_preceding_pairs.add(pair)
                                self.continuation_counts[word] += 1
        return time.time() - t0

    def get_probability(self, word, context):
        context = tuple(context)
        v_size = len(self.vocab)
        c_ngram = self.ngram_counts.get(context + (word,), 0)
        c_context = self.context_counts.get(context, 0)
        
        if self.smoothing == "laplace":
            return (c_ngram + 1) / (c_context + v_size)
        elif self.smoothing == "add_k":
            return (c_ngram + self.k) / (c_context + (self.k * v_size))
        elif self.smoothing == "kn":
            p_continuation = self.continuation_counts.get(word, 0) / max(1, len(self.seen_preceding_pairs))
            if p_continuation == 0:
                p_continuation = 1.0 / max(1, v_size)
            if c_context == 0:
                return p_continuation
            unique_follows = self.context_unique_follows.get(context, 0)
            lambda_ctx = (self.d / c_context) * unique_follows
            return max(c_ngram - self.d, 0) / c_context + (lambda_ctx * p_continuation)

    def evaluate_ppl(self, test_path, max_sentences=500):
        with open(test_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if len(line.strip()) > 30][:max_sentences]
            
        total_log_prob = 0.0
        total_tokens = 0
        word_token = re.compile(r'\b\w+\b')
        sent_split = re.compile(r'[.!?]+')

        for line in lines:
            for sent in sent_split.split(line):
                raw_tokens = word_token.findall(sent.lower())
                if len(raw_tokens) < self.n:
                    continue
                tokens = ['<BOS>'] * (self.n - 1) + raw_tokens + ['<EOS>']
                for i in range(self.n - 1, len(tokens)):
                    context = tuple(tokens[i - (self.n - 1):i])
                    target = tokens[i]
                    prob = self.get_probability(target, context)
                    total_log_prob += math.log2(prob)
                    total_tokens += 1

        if total_tokens == 0:
            return float('inf')
        return math.pow(2, -(total_log_prob / total_tokens))

if __name__ == "__main__":
    n_options = [2, 3, 4]
    smoothing_options = ["laplace", "add_k", "kn"]

    print("="*65)
    print("  [1/3] JALANKAN EKSPERIMEN TRAINING & SIMPAN MODEL (.PKL)")
    print("="*65)

    for n in n_options:
        for sm in smoothing_options:
            pkl_file = f"ngram_model_N{n}_{sm}.pkl"
            if os.path.exists(pkl_file):
                print(f"[*] Cache '{pkl_file}' sudah ada, melewatin proses training.")
                continue

            print(f"[>] Training: N={n} | Smoothing={sm.upper()}...")
            model = ComprehensiveNGram(n=n, smoothing=sm)
            t_dur = model.train(TRAIN_FILE)
            
            with open(pkl_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"[✓] Berhasil disimpan ke '{pkl_file}' ({t_dur:.2f}s)")