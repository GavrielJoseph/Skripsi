"""
fold_analysis.py — Analisis Confusion Matrix per Fold (5-Fold Cross Validation)
Load data dan preprocessing IDENTIK dengan train.py supaya jumlah data sama.

Cara menjalankan:
  python fold_analysis.py

Output:
  - 5 confusion matrix per fold di terminal
  - Rata-rata confusion matrix
  - Hasil disimpan ke backend/ml/models/fold_analysis.json
"""

import os
import re
import json
import joblib
import warnings
import requests
import numpy as np
import pandas as pd

from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from collections import Counter

warnings.filterwarnings("ignore")

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    STEMMER_AVAILABLE = True
except ImportError:
    STEMMER_AVAILABLE = False
    print("[WARNING] PySastrawi tidak ada.")

# ── Path — sama persis dengan train.py ───────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
TRAINING_CSV = os.path.join(DATA_DIR,  "training_data.csv")
KAMUS_PATH   = os.path.join(DATA_DIR,  "kamus_alay.json")
BIGRAM_PATH  = os.path.join(MODEL_DIR, "bigram_phraser.pkl")
W2V_PATH     = os.path.join(MODEL_DIR, "vectorizer.pkl")
G2G_CSV      = os.path.join(DATA_DIR,  "G2G_dataset.csv")
HEYXI_CSV    = os.path.join(DATA_DIR,  "Heyxi_dataset.csv")
OUTPUT_PATH  = os.path.join(MODEL_DIR, "fold_analysis.json")

# ── Stopwords — identik dengan train.py ──────────────────────────────────────
CUSTOM_STOPWORDS = {
    "nya","yg","aja","jd","tp","banget","sih","buat","biar","mah",
    "kalo","pas","terus","sama","udah","gini","gitu","ya","deh","dong",
    "kok","kan","pake","sekali","baru","dari","ke","di","ini","itu",
    "yang","dan","aku","saya","dia","mereka","kita","kami","untuk",
    "dalam","pada","juga","sudah","ada","saja","lagi","karena",
    "kalau","pun","bisa","akan","jadi","tapi"
}
NEGASI_WORDS = {"tidak","bukan","jangan","belum","tanpa","kurang"}
STOPWORDS_ID = set(stopwords.words("indonesian")).union(CUSTOM_STOPWORDS) - NEGASI_WORDS

EMOJI_MAP = {
    "👍":" bagus ","👎":" buruk ","😍":" suka ","🥰":" suka ","😘":" suka ",
    "❤️":" suka ","💕":" suka ","💞":" suka ","😊":" senang ","😁":" senang ",
    "🙏":" terima kasih ","😭":" kecewa ","😢":" kecewa ","😡":" marah ",
    "🤢":" buruk ","🤮":" buruk ","⭐":" bintang ","🔥":" bagus ","✨":" bagus ",
}

# ── Kamus skincare — identik dengan train.py ─────────────────────────────────
KAMUS_SKINCARE = {
    "tak":"tidak","gak":"tidak","ga":"tidak","ngga":"tidak","nggak":"tidak",
    "tdk":"tidak","gk":"tidak","ngak":"tidak","kagak":"tidak",
    "bgt":"banget","bgtt":"banget","bngt":"banget","bangettt":"banget",
    "yg":"yang","dg":"dengan","dgn":"dengan","utk":"untuk",
    "udh":"sudah","udah":"sudah","dah":"sudah",
    "blm":"belum","belom":"belum","jg":"juga","jga":"juga",
    "krn":"karena","karna":"karena","klo":"kalau","kalo":"kalau",
    "aja":"saja","doang":"saja","gue":"saya","gw":"saya","sy":"saya",
    "lo":"kamu","lu":"kamu","pake":"pakai","pakek":"pakai",
    "dipake":"dipakai","emang":"memang","emg":"memang",
    "msh":"masih","tetep":"tetap","gabisa":"tidak bisa","gbs":"tidak bisa",
    "sampe":"sampai","nyampe":"sampai","smua":"semua",
    "ngilangin":"menghilangkan","ilangin":"menghilangkan","ilang":"hilang",
    "ngurangin":"mengurangi","ngatasin":"mengatasi",
    "nyembuhin":"menyembuhkan","mudar":"memudar","pudar":"memudar",
    "berkurang":"berkurang","nyerap":"menyerap","meresap":"meresap",
    "ngelembapkan":"melembapkan","lembapkan":"melembapkan",
    "ngelebabpin":"melembapkan","ngebantu":"membantu","bantu":"membantu",
    "ngerasa":"merasa","kerasa":"terasa","berasa":"terasa",
    "bruntuaan":"berjerawat","bruntusan":"berjerawat","bruntus":"berjerawat",
    "jerawatan":"berjerawat","gatel":"gatal","lembab":"lembap",
    "minyakan":"berminyak","glowing":"bercahaya","pori2":"pori","sensitip":"sensitif",
    "bestt":"bagus","best":"bagus","baguss":"bagus","bagusss":"bagus",
    "cocokk":"cocok","mantap":"bagus","mantul":"bagus","mantab":"bagus",
    "kece":"bagus","oke":"baik","ok":"baik","jos":"bagus","joss":"bagus",
    "jelek":"buruk","ancur":"buruk",
    "rekomen":"rekomendasi","recommended":"rekomendasi",
    "ampuh":"efektif","manjur":"efektif",
    "worth":"sepadan","worthit":"sepadan",
    "murah":"murah","murce":"murah","murmer":"murah",
    "muka":"wajah","muke":"wajah","jerawat":"jerawat","jrawat":"jerawat",
    "iritasi":"iritasi","skincare":"skincare","skincr":"skincare",
    "moisturizer":"pelembap","mois":"pelembap",
    "sunscreen":"tabir surya","sesuwai":"sesuai",
    "ori":"original","asli":"original","palsu":"palsu","kw":"palsu",
    "packaging":"kemasan","packing":"kemasan",
    "makasih":"terima kasih","tq":"terima kasih",
    "alhamdulillah":"bagus","masyaallah":"bagus",
    "sih":"","deh":"","dong":"","loh":"","wkwk":"","hehe":"","haha":"",
}


# ── load kamus — identik dengan train.py ─────────────────────────────────────
def load_kamus_alay():
    if os.path.exists(KAMUS_PATH):
        with open(KAMUS_PATH, "r", encoding="utf-8") as f:
            kamus = json.load(f)
        if "tak" not in kamus:
            kamus["tak"] = "tidak"
        return kamus
    # kalau belum ada, pakai kamus skincare saja
    print("[WARNING] kamus_alay.json tidak ditemukan, pakai kamus skincare saja.")
    return KAMUS_SKINCARE.copy()

KAMUS_ALAY = load_kamus_alay()


# ── preprocessing — identik dengan train.py ───────────────────────────────────
def translate_emoji(text):
    for emoji, word in EMOJI_MAP.items():
        text = text.replace(emoji, word)
    return text

def clean_repeated(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def normalize_alay(text):
    return " ".join([v for w in text.split() for v in [KAMUS_ALAY.get(w, w)] if v])

def handle_negation(tokens):
    result, i = [], 0
    while i < len(tokens):
        if tokens[i] in NEGASI_WORDS and i + 1 < len(tokens):
            result.append(f"tidak_{tokens[i+1]}")
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

def is_noise(text):
    if not isinstance(text, str):
        return True
    t = text.strip()
    if len(t) < 5:
        return True
    ascii_only = t.encode("ascii", "ignore").decode("ascii").strip()
    alpha = re.sub(r'[^a-zA-Z]', '', ascii_only)
    if len(alpha) < 3:
        return False if len(t) >= 20 else True
    if len(set(alpha.lower())) / len(alpha) < 0.15:
        return True
    words = ascii_only.split()
    if words and any(len(w) > 25 for w in words):
        return True
    return False

def preprocess_text(text):
    if not isinstance(text, str) or len(text.strip()) < 3:
        return []
    text = translate_emoji(text)
    text = clean_repeated(text)
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = normalize_alay(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) >= 2]
    tokens = handle_negation(tokens)
    if STEMMER_AVAILABLE:
        tokens = [t if t.startswith("tidak_") else stemmer.stem(t) for t in tokens]
    return tokens


# ── load dataset — identik dengan train.py ───────────────────────────────────
def load_sekarmulyani():
    if not os.path.exists(TRAINING_CSV):
        raise FileNotFoundError(
            f"training_data.csv tidak ditemukan: {TRAINING_CSV}\n"
            "Jalankan train.py dulu untuk membuat cache dataset."
        )
    df = pd.read_csv(TRAINING_CSV)
    if "source" not in df.columns:
        df["source"] = "sekarmulyani"
    return df

def load_csv_domain(csv_path, source_name):
    if not os.path.exists(csv_path):
        print(f"[WARNING] {source_name} tidak ditemukan: {csv_path}")
        return pd.DataFrame(columns=["review_text","label","source"])
    df = None
    for sep in [",",";"]:
        try:
            tmp = pd.read_csv(csv_path, sep=sep, encoding="utf-8-sig")
            if len(tmp.columns) > 1:
                df = tmp
                break
        except:
            continue
    if df is None:
        return pd.DataFrame(columns=["review_text","label","source"])
    rows = []
    for _, row in df.iterrows():
        text   = str(row.get("Comment","")).strip()
        rating = row.get("Rating", 0)
        if not text or len(text) < 5:
            continue
        try:
            rating = float(rating)
        except:
            continue
        if   rating <= 2: label = "negative"
        elif rating >= 4: label = "positive"
        else: continue
        rows.append({"review_text": text, "label": label, "source": source_name})
    return pd.DataFrame(rows).drop_duplicates(subset=["review_text"])

def get_document_vector(tokens, w2v_model):
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ANALISIS CONFUSION MATRIX PER FOLD (5-Fold CV)")
    print("  Parameter: C=10.0, class_weight='balanced'")
    print("  Data & preprocessing: IDENTIK dengan train.py")
    print("=" * 60)

    # ── 1. Load data — sama persis dengan train.py ───────────────
    print("\n[STEP 1] Load dataset...")
    df = pd.concat([
        load_sekarmulyani(),
        load_csv_domain(G2G_CSV,   "g2g_tokopedia"),
        load_csv_domain(HEYXI_CSV, "heyxi_tokopedia"),
    ], ignore_index=True)

    df = df.dropna(subset=["review_text","label"])
    df = df[df["label"].isin(["positive","negative"])]
    df = df.drop_duplicates(subset=["review_text"])

    before = len(df)
    df = df[~df["review_text"].apply(is_noise)]
    print(f"[INFO] Noise dibuang: {before - len(df)} baris")
    print(f"[INFO] Total data bersih: {len(df)} ulasan")

    # ── 2. Preprocessing ─────────────────────────────────────────
    print("\n[STEP 2] Preprocessing...")
    df["tokens"] = df["review_text"].apply(preprocess_text)

    # ── 3. Load bigram phraser ────────────────────────────────────
    phraser = joblib.load(BIGRAM_PATH) if os.path.exists(BIGRAM_PATH) else None
    if phraser:
        df["tokens"] = df["tokens"].apply(lambda t: list(phraser[t]))

    df = df[df["tokens"].map(len) > 0].reset_index(drop=True)
    print(f"[INFO] Setelah preprocessing: {len(df)} ulasan")

    # ── 4. Load Word2Vec & vectorize ─────────────────────────────
    print("\n[STEP 3] Vectorizing...")
    if not os.path.exists(W2V_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {W2V_PATH}\nJalankan train.py dulu.")
    w2v_model = joblib.load(W2V_PATH)
    X = np.array([get_document_vector(t, w2v_model) for t in df["tokens"]])
    y = df["label"].values
    print(f"[INFO] Shape X: {X.shape}")

    # ── 5. 5-Fold CV ─────────────────────────────────────────────
    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print(f"\n{'='*60}")
    print("  HASIL PER FOLD")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        lr = LogisticRegression(
            C=10.0, class_weight="balanced",
            max_iter=1000, solver="lbfgs", random_state=42
        )
        lr.fit(X_tr, y_tr)
        y_pred = lr.predict(X_val)

        classes          = ["negative","positive"]
        cm               = confusion_matrix(y_val, y_pred, labels=classes)
        tn, fp, fn, tp   = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

        acc = round(accuracy_score(y_val, y_pred) * 100, 2)
        pre = round(precision_score(y_val, y_pred, average="weighted") * 100, 2)
        rec = round(recall_score(y_val, y_pred, average="weighted") * 100, 2)
        f1  = round(f1_score(y_val, y_pred, average="weighted") * 100, 2)

        results.append({
            "fold": fold, "n_train": len(X_tr), "n_val": len(X_val),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": pre, "recall": rec, "f1_score": f1,
        })

        print(f"\n  Fold {fold}:")
        print(f"  Train={len(X_tr)} | Validasi={len(X_val)}")
        print(f"                 | Pred Negatif | Pred Positif")
        print(f"  Aktual Negatif | {tn:12,} | {fp:12,}")
        print(f"  Aktual Positif | {fn:12,} | {tp:12,}")
        print(f"  Accuracy={acc}%  Precision={pre}%  Recall={rec}%  F1={f1}%")

    # ── 6. Rata-rata ─────────────────────────────────────────────
    avg = {
        "tp":           round(float(np.mean([r["tp"]        for r in results])), 2),
        "tn":           round(float(np.mean([r["tn"]        for r in results])), 2),
        "fp":           round(float(np.mean([r["fp"]        for r in results])), 2),
        "fn":           round(float(np.mean([r["fn"]        for r in results])), 2),
        "accuracy":     round(float(np.mean([r["accuracy"]  for r in results])), 2),
        "precision":    round(float(np.mean([r["precision"] for r in results])), 2),
        "recall":       round(float(np.mean([r["recall"]    for r in results])), 2),
        "f1_score":     round(float(np.mean([r["f1_score"]  for r in results])), 2),
        "std_accuracy": round(float(np.std( [r["accuracy"]  for r in results])), 4),
    }

    print(f"\n{'='*60}")
    print("  RATA-RATA 5 FOLD")
    print(f"{'='*60}")
    print(f"                 | Pred Negatif | Pred Positif")
    print(f"  Aktual Negatif | {avg['tn']:12} | {avg['fp']:12}")
    print(f"  Aktual Positif | {avg['fn']:12} | {avg['tp']:12}")
    print(f"\n  Accuracy  : {avg['accuracy']}% (±{avg['std_accuracy']}%)")
    print(f"  Precision : {avg['precision']}%")
    print(f"  Recall    : {avg['recall']}%")
    print(f"  F1-Score  : {avg['f1_score']}%")

    # ── 7. Simpan ────────────────────────────────────────────────
    output = {
        "keterangan": (
            "Confusion matrix per fold dari 5-fold cross validation "
            "menggunakan parameter terbaik (C=10.0, class_weight=balanced). "
            "Data dan preprocessing identik dengan train.py."
        ),
        "total_data":  len(df),
        "parameter":   {"C": 10.0, "class_weight": "balanced"},
        "folds":       results,
        "average":     avg,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Hasil disimpan ke: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()