"""
train.py
--------
Jalankan file ini SATU KALI untuk melatih model.
Hasil training disimpan di folder: ml/models/

Cara menjalankan (dari folder ml/):
    python scripts/train.py

Output yang dihasilkan:
    models/word2vec.model
    models/logistic_regression.pkl
    models/vectorizer.pkl
    models/model_performance.json   <- hasil evaluasi model
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import nltk

from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────
# 1. KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_CSV = os.path.join(DATA_DIR, "training_data.csv")

# ─────────────────────────────────────────────────────────────
# 2. NLTK & STEMMER SETUP
# ─────────────────────────────────────────────────────────────

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
STOPWORDS_ID = set(stopwords.words("indonesian"))

# Stemmer — pakai PySastrawi kalau ada, fallback ke tanpa stemming
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    STEMMER_AVAILABLE = True
    print("[INFO] PySastrawi stemmer loaded.")
except ImportError:
    STEMMER_AVAILABLE = False
    print("[WARNING] PySastrawi tidak terinstall. Jalankan: pip install PySastrawi")
    print("[WARNING] Training tetap lanjut tanpa stemming.")

# ─────────────────────────────────────────────────────────────
# 3. KAMUS NORMALISASI KATA ALAY/GAUL
# ─────────────────────────────────────────────────────────────

# Kamus dasar kata alay → formal
# Referensi: github.com/nasalsabila/kamus-alay
KAMUS_ALAY = {
    "gak": "tidak", "ga": "tidak", "ngga": "tidak", "nggak": "tidak",
    "tdk": "tidak", "g": "tidak", "gk": "tidak", "ngak": "tidak",
    "bgt": "banget", "bgtt": "banget", "bngt": "banget",
    "yg": "yang", "yng": "yang", "dg": "dengan", "dgn": "dengan",
    "utk": "untuk", "tuk": "untuk", "udh": "sudah", "udah": "sudah",
    "sdh": "sudah", "blm": "belum", "blum": "belum",
    "jg": "juga", "juga2": "juga", "krn": "karena", "karna": "karena",
    "klo": "kalau", "kalo": "kalau", "kal": "kalau",
    "sm": "sama", "ama": "sama", "emg": "memang", "emang": "memang",
    "msh": "masih", "msih": "masih", "bisa2": "bisa",
    "aja": "saja", "aj": "saja", "doang": "saja",
    "gue": "saya", "gw": "saya", "ane": "saya", "w": "saya",
    "lo": "kamu", "lu": "kamu", "elu": "kamu",
    "beli2": "beli", "pake": "pakai", "dipake": "dipakai",
    "cocok2": "cocok", "jelek2": "jelek",
    "rekomen": "rekomendasi", "recomen": "rekomendasi",
    "mantap": "bagus", "mantul": "bagus", "mantab": "bagus",
    "kece": "bagus", "oke": "baik", "okey": "baik", "ok": "baik",
    "jos": "bagus", "joss": "bagus", "top": "bagus",
    "jelek": "buruk", "jlek": "buruk", "ancur": "buruk",
    "muka": "wajah", "muke": "wajah",
    "lembab": "lembap", "lembabb": "lembap",
    "iritasi": "iritasi", "irit": "iritasi",
    "jerawat": "jerawat", "jrawat": "jerawat",
    "spf": "spf", "skincare": "skincare", "skincr": "skincare",
    "moisturizer": "pelembap", "serum": "serum",
    "ngaruh": "berpengaruh", "berasa": "terasa",
    "nyaman": "nyaman", "enak": "nyaman",
    "ampuh": "efektif", "manjur": "efektif",
    "murah": "murah", "mahal": "mahal",
    "recommended": "rekomendasi", "recommend": "rekomendasi",
    "worth": "sepadan", "worthit": "sepadan",
}


def normalize_alay(text: str) -> str:
    """Ganti kata alay/gaul ke bentuk formal."""
    words = text.split()
    result = []
    for word in words:
        result.append(KAMUS_ALAY.get(word, word))
    return " ".join(result)


# ─────────────────────────────────────────────────────────────
# 4. LOAD DATASET
# ─────────────────────────────────────────────────────────────

def load_training_data():
    if os.path.exists(TRAINING_CSV):
        print(f"[INFO] Memuat dataset lokal: {TRAINING_CSV}")
        df = pd.read_csv(TRAINING_CSV)
        if "review_text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV harus punya kolom 'review_text' dan 'label'")
        return df

    print("[INFO] Mendownload dataset: sekarmulyani/ulasan-beauty-products")
    print("[INFO] Mengambil split: train + validation + test (total ~76.000 baris)")

    dataset = load_dataset("sekarmulyani/ulasan-beauty-products")

    frames = []
    for split_name in ["train", "validation", "test"]:
        if split_name in dataset:
            frames.append(dataset[split_name].to_pandas())
            print(f"[INFO] Split '{split_name}': {len(dataset[split_name])} baris")

    raw = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Total sebelum filter: {len(raw)} baris")

    # Konversi one-hot rating → label sentimen
    # Bintang 1 & 2 → negative
    # Bintang 3     → buang (ambigu)
    # Bintang 4 & 5 → positive
    rows = []
    for _, row in raw.iterrows():
        text = str(row.get("Review", "")).strip()
        if not text:
            continue

        if row.get("Bintang 1") or row.get("Bintang 2"):
            label = "negative"
        elif row.get("Bintang 4") or row.get("Bintang 5"):
            label = "positive"
        else:
            continue  # buang bintang 3

        rows.append({"review_text": text, "label": label})

    df = pd.DataFrame(rows)
    df.to_csv(TRAINING_CSV, index=False)
    print(f"[INFO] Dataset disimpan ke: {TRAINING_CSV}")
    print(f"[INFO] Total setelah filter rating 3: {len(df)} baris")
    print(f"[INFO] Distribusi:\n{df['label'].value_counts().to_string()}")

    return df


# ─────────────────────────────────────────────────────────────
# 5. PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> list:
    if not isinstance(text, str):
        return []

    # Lowercase
    text = text.lower()

    # Hapus URL, mention, hashtag
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)

    # Hapus emoji dan karakter non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")

    # Hapus tanda baca dan angka
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalisasi kata alay
    text = normalize_alay(text)

    # Tokenisasi
    tokens = text.split()

    # Hapus stopwords dan token pendek
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) >= 2]

    # Stemming
    if STEMMER_AVAILABLE:
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


# ─────────────────────────────────────────────────────────────
# 6. WORD2VEC
# ─────────────────────────────────────────────────────────────

def train_word2vec(corpus):
    print("[INFO] Melatih Word2Vec...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,       # skip-gram
        workers=4,
        epochs=10,
        seed=42
    )
    path = os.path.join(MODEL_DIR, "word2vec.model")
    model.save(path)
    print(f"[INFO] Word2Vec disimpan: {path}")
    return model


def get_document_vector(tokens, w2v_model):
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)


# ─────────────────────────────────────────────────────────────
# 7. LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train):
    print("[INFO] Melatih Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_train, y_train)
    path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
    joblib.dump(model, path)
    print(f"[INFO] Model disimpan: {path}")
    return model


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  TRAINING PIPELINE — Skincare Sentiment Analysis")
    print("=" * 55)

    # Step 1: Load data
    print("\n[STEP 1] Load dataset...")
    df = load_training_data()
    df = df.dropna(subset=["review_text", "label"])
    df = df[df["label"].isin(["positive", "negative"])]
    print(f"[INFO] Total data bersih: {len(df)} baris")
    print(f"[INFO] Distribusi:\n{df['label'].value_counts().to_string()}")

    # Step 2: Preprocessing
    print("\n[STEP 2] Preprocessing teks...")
    df["tokens"] = df["review_text"].apply(preprocess_text)
    df = df[df["tokens"].map(len) > 0]
    print(f"[INFO] Setelah preprocessing: {len(df)} baris")

    # Step 3: Word2Vec
    print("\n[STEP 3] Word2Vec embedding...")
    corpus = df["tokens"].tolist()
    w2v_model = train_word2vec(corpus)
    joblib.dump(w2v_model, os.path.join(MODEL_DIR, "vectorizer.pkl"))

    # Step 4: Document vectors
    print("\n[STEP 4] Membuat document vectors...")
    X = np.array([get_document_vector(t, w2v_model) for t in df["tokens"]])
    y = df["label"].values
    print(f"[INFO] Shape: {X.shape}")

    # Step 5: Split
    print("\n[STEP 5] Split 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Training: {len(X_train)} | Testing: {len(X_test)}")

    # Step 6: Train
    print("\n[STEP 6] Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)

    # Step 7: Evaluasi
    print("\n[STEP 7] Evaluasi model:")
    y_pred = lr_model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="positive")
    recall    = recall_score(y_test, y_pred, pos_label="positive")
    f1        = f1_score(y_test, y_pred, pos_label="positive")

    print(f"\n  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall    : {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    # Step 8: Simpan hasil evaluasi ke JSON
    performance = {
        "accuracy":       round(accuracy * 100, 2),
        "precision":      round(precision * 100, 2),
        "recall":         round(recall * 100, 2),
        "f1_score":       round(f1 * 100, 2),
        "training_size":  int(len(X_train)),
        "testing_size":   int(len(X_test)),
        "total_data":     int(len(df)),
        "dataset":        "sekarmulyani/ulasan-beauty-products",
        "algorithm":      "Word2Vec + Logistic Regression",
    }

    perf_path = os.path.join(MODEL_DIR, "model_performance.json")
    with open(perf_path, "w") as f:
        json.dump(performance, f, indent=2)
    print(f"\n[INFO] Hasil evaluasi disimpan: {perf_path}")

    print("\n" + "=" * 55)
    print("  TRAINING SELESAI!")
    print(f"  Model disimpan di: {MODEL_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()