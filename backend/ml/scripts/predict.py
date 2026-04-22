"""
predict.py
----------
Dipanggil oleh Laravel setiap kali user upload CSV untuk dianalisa.

Cara Laravel memanggil (dari controller PHP):
    python ml/scripts/predict.py /path/to/uploaded_file.csv

Output ke STDOUT dalam format JSON — Laravel membaca ini.
"""

import os
import re
import sys
import json
import joblib
import warnings
import pandas as pd
import numpy as np
import nltk

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import Counter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LR_MODEL_PATH  = os.path.join(MODEL_DIR, "logistic_regression.pkl")
W2V_MODEL_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
STOPWORDS_ID = set(stopwords.words("indonesian"))

# Stemmer
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    STEMMER_AVAILABLE = True
except ImportError:
    STEMMER_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# KAMUS NORMALISASI KATA ALAY/GAUL
# Referensi: github.com/nasalsabila/kamus-alay
# ─────────────────────────────────────────────────────────────

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


def normalize_alay(text):
    words = text.split()
    return " ".join([KAMUS_ALAY.get(w, w) for w in words])


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# Urutan: Cleaning → Case Folding → Normalisasi Alay →
#         Tokenizing → Stopword Removal → Stemming
# ─────────────────────────────────────────────────────────────

def preprocess_text(text):
    if not isinstance(text, str):
        return []

    # 1. Cleaning — hapus URL, mention, hashtag, emoji, tanda baca
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 2. Case Folding — semua jadi lowercase
    text = text.lower()

    # 3. Normalisasi kata alay/gaul → formal
    text = normalize_alay(text)

    # 4. Tokenizing — pisah jadi kata-kata
    tokens = text.split()

    # 5. Stopword Removal — hapus kata tidak bermakna
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) >= 2]

    # 6. Stemming — kata berimbuhan → kata dasar
    if STEMMER_AVAILABLE:
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def get_document_vector(tokens, w2v_model):
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

def load_models():
    if not os.path.exists(LR_MODEL_PATH):
        raise FileNotFoundError(
            f"Model belum ada: {LR_MODEL_PATH}\n"
            "Jalankan dulu: python scripts/train.py"
        )
    if not os.path.exists(W2V_MODEL_PATH):
        raise FileNotFoundError(
            f"Word2Vec model belum ada: {W2V_MODEL_PATH}\n"
            "Jalankan dulu: python scripts/train.py"
        )
    lr_model  = joblib.load(LR_MODEL_PATH)
    w2v_model = joblib.load(W2V_MODEL_PATH)
    return lr_model, w2v_model


# ─────────────────────────────────────────────────────────────
# LOAD CSV — auto-detect separator ; atau ,
# ─────────────────────────────────────────────────────────────

def load_csv(csv_path):
    for sep in [";", ","]:
        try:
            tmp = pd.read_csv(csv_path, sep=sep, encoding="utf-8-sig")
            if len(tmp.columns) > 1:
                return tmp
        except Exception:
            continue
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def find_text_column(df):
    col_map = {c.lower(): c for c in df.columns}
    for candidate in ["comment", "review_text", "review", "ulasan", "text", "komentar"]:
        if candidate in col_map:
            return col_map[candidate]
    return df.columns[0]


def find_product_column(df):
    col_map = {c.lower(): c for c in df.columns}
    for candidate in ["productname", "product_name", "product", "produk", "nama_produk"]:
        if candidate in col_map:
            return col_map[candidate]
    return None


# ─────────────────────────────────────────────────────────────
# CLUSTERING — Agglomerative Hierarchical
# ─────────────────────────────────────────────────────────────

def find_optimal_clusters(X, max_k=6):
    if len(X) < 4:
        return 2
    best_k     = 2
    best_score = -1
    for k in range(2, min(max_k + 1, len(X))):
        clustering = AgglomerativeClustering(n_clusters=k, linkage="average")
        labels = clustering.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k     = k
    return best_k


def run_clustering(X, n_clusters):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
    return clustering.fit_predict(X)


# ─────────────────────────────────────────────────────────────
# KEYWORD EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_keywords_per_doc(tokens, top_n=5):
    freq = Counter(tokens)
    return [word for word, _ in freq.most_common(top_n)]


def extract_keywords_per_cluster(cluster_id, all_tokens, cluster_labels, top_n=10):
    cluster_tokens = []
    for tokens, label in zip(all_tokens, cluster_labels):
        if label == cluster_id:
            cluster_tokens.extend(tokens)
    if not cluster_tokens:
        return []
    freq = Counter(cluster_tokens)
    return [word for word, _ in freq.most_common(top_n)]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        sys.stdout.buffer.write(json.dumps({
            "status": "error",
            "message": "Tidak ada file CSV yang diberikan."
        }, ensure_ascii=False).encode("utf-8"))
        sys.exit(1)

    csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        sys.stdout.buffer.write(json.dumps({
            "status": "error",
            "message": f"File tidak ditemukan: {csv_path}"
        }, ensure_ascii=False).encode("utf-8"))
        sys.exit(1)

    # Load CSV
    try:
        df = load_csv(csv_path)
    except Exception as e:
        sys.stdout.buffer.write(json.dumps({
            "status": "error",
            "message": f"Gagal membaca CSV: {str(e)}"
        }, ensure_ascii=False).encode("utf-8"))
        sys.exit(1)

    # Tentukan kolom
    text_column    = find_text_column(df)
    product_column = find_product_column(df)

    df["review_text"]  = df[text_column].astype(str)
    df["product_name"] = df[product_column].astype(str) if product_column else ""
    df = df[df["review_text"].str.strip() != ""]
    df = df.dropna(subset=["review_text"])
    df = df.reset_index(drop=True)

    if len(df) == 0:
        sys.stdout.buffer.write(json.dumps({
            "status": "error",
            "message": "CSV tidak punya data teks ulasan."
        }, ensure_ascii=False).encode("utf-8"))
        sys.exit(1)

    # Load model
    try:
        lr_model, w2v_model = load_models()
    except FileNotFoundError as e:
        sys.stdout.buffer.write(json.dumps({
            "status": "error",
            "message": str(e)
        }, ensure_ascii=False).encode("utf-8"))
        sys.exit(1)

    # ── Preprocessing ──────────────────────────────────────────
    # Urutan: Cleaning → Case Folding → Normalisasi Alay →
    #         Tokenizing → Stopword Removal → Stemming
    df["tokens"] = df["review_text"].apply(preprocess_text)

    # ── Word2Vec → Document Vectors ───────────────────────────
    X = np.array([get_document_vector(t, w2v_model) for t in df["tokens"]])

    # ── Sentiment Classification + Confidence Score ───────────
    df["sentiment"]  = lr_model.predict(X)
    proba            = lr_model.predict_proba(X)
    classes          = list(lr_model.classes_)
    df["confidence"] = [
        round(float(proba[i][classes.index(df["sentiment"][i])]), 4)
        for i in range(len(df))
    ]

    # ── Agglomerative Hierarchical Clustering ─────────────────
    n_clusters     = find_optimal_clusters(X)
    cluster_labels = run_clustering(X, n_clusters)
    df["cluster_id"] = cluster_labels.tolist()

    # ── Keyword per dokumen ────────────────────────────────────
    df["keywords"] = df["tokens"].apply(
        lambda t: extract_keywords_per_doc(t, top_n=5)
    )

    # ── Keyword per cluster ────────────────────────────────────
    cluster_keywords = {}
    for cid in range(n_clusters):
        kw    = extract_keywords_per_cluster(
            cid, df["tokens"].tolist(), cluster_labels, top_n=10
        )
        count = int((cluster_labels == cid).sum())
        cluster_keywords[str(cid)] = {
            "label":        f"Cluster {cid}",
            "count":        count,
            "top_keywords": kw
        }

    # ── Summary ────────────────────────────────────────────────
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    avg_confidence   = round(float(df["confidence"].mean()), 4)

    # ── Output JSON ────────────────────────────────────────────
    results = []
    for _, row in df.iterrows():
        results.append({
            "product_name": row["product_name"],
            "review_text":  row["review_text"],
            "sentiment":    row["sentiment"],
            "confidence":   row["confidence"],
            "cluster_id":   int(row["cluster_id"]),
            "keywords":     row["keywords"]
        })

    output = {
        "status":  "success",
        "total":   len(df),
        "results": results,
        "summary": {
            "positive":        int(sentiment_counts.get("positive", 0)),
            "negative":        int(sentiment_counts.get("negative", 0)),
            "avg_confidence":  avg_confidence,
            "n_clusters":      n_clusters,
            "clusters":        cluster_keywords
        }
    }

    sys.stdout.buffer.write(
        json.dumps(output, ensure_ascii=False).encode("utf-8")
    )


if __name__ == "__main__":
    main()