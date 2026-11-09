import xml.etree.ElementTree as ET
import re
import random
import time

random.seed(42)

def clean_wikitext(text):
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref.*?>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref.*?>', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\[(Kategori|Berkas|File|Gambar):.*?\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'={2,}.*?={2,}', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.,!\?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_and_split_corpus(xml_path, sample_rate=0.20, train_ratio=0.8):
    start_time = time.time()
    extracted_lines = []
    
    print(f"[*] Parsing {xml_path} (Sampling {sample_rate*100}%)...")
    for event, elem in ET.iterparse(xml_path, events=('end',)):
        if elem.tag.endswith('page'):
            ns = elem.find('{*}ns')
            if ns is not None and ns.text == '0':
                if random.random() < sample_rate:
                    revision = elem.find('{*}revision')
                    if revision is not None:
                        text_elem = revision.find('{*}text')
                        if text_elem is not None and text_elem.text:
                            raw_text = text_elem.text
                            if not raw_text.lstrip().startswith(('#ALIH', '#REDIRECT')):
                                cleaned = clean_wikitext(raw_text)
                                if len(cleaned) > 50:
                                    extracted_lines.append(cleaned)
            elem.clear()

    print(f"[✓] Ekstraksi selesai: {len(extracted_lines):,} artikel.")
    
    # Shuffle & Split 80:20
    random.shuffle(extracted_lines)
    split_idx = int(len(extracted_lines) * train_ratio)
    train_data = extracted_lines[:split_idx]
    test_data = extracted_lines[split_idx:]

    print(f"[*] Menyimpan Data Train ({len(train_data):,} artikel) ke 'train_corpus.txt'...")
    with open("train_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))

    print(f"[*] Menyimpan Data Test ({len(test_data):,} artikel) ke 'test_corpus.txt'...")
    with open("test_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test_data))

    print(f"[✓] Pemisahan Dataset Selesai ({time.time() - start_time:.2f}s)")

if __name__ == "__main__":
    # Ganti 'korpus.xml' sesuai file XML-mu
    process_and_split_corpus("idwiki-latest-pages-articles.xml", sample_rate=0.05, train_ratio=0.80)