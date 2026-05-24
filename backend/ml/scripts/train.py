import os, re, json, joblib, requests
import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix
)
from datasets import load_dataset
from collections import Counter

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

TRAINING_CSV = os.path.join(DATA_DIR,  "training_data.csv")
KAMUS_PATH   = os.path.join(DATA_DIR,  "kamus_alay.json")
BIGRAM_PATH  = os.path.join(MODEL_DIR, "bigram_phraser.pkl")
G2G_CSV      = os.path.join(DATA_DIR,  "G2G_dataset.csv")
HEYXI_CSV    = os.path.join(DATA_DIR,  "Heyxi_dataset.csv")

nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# Custom stopwords untuk bahasa gaul e-commerce Indonesia
# Catatan: "tidak" TIDAK dimasukkan di sini karena dilindungi oleh NEGASI_WORDS
CUSTOM_STOPWORDS = {
    "nya", "yg", "aja", "jd", "tp", "banget", "sih", "buat", "biar", "mah",
    "kalo", "pas", "terus", "sama", "udah", "gini", "gitu", "ya", "deh", "dong",
    "kok", "kan", "pake", "sekali", "baru", "dari", "ke", "di", "ini", "itu",
    "yang", "dan", "aku", "saya", "dia", "mereka", "kita", "kami", "untuk",
    "dalam", "pada", "juga", "sudah", "ada", "saja", "lagi", "karena",
    "kalau", "pun", "bisa", "akan", "jadi", "tapi"
}
# FIX: "tidak" dihapus dari CUSTOM_STOPWORDS karena ada di NEGASI_WORDS
# Sebelumnya duplikat di dua set — tidak broken tapi membingungkan secara logika

NEGASI_WORDS = {"tidak", "bukan", "jangan", "belum", "tanpa", "kurang"}
# STOPWORDS_ID = gabungan stopwords bawaan NLTK + custom, DIKURANGI kata negasi
# Kata negasi WAJIB dipertahankan karena mengubah makna sentimen:
# "tidak cocok" ≠ "cocok", "tidak bagus" ≠ "bagus"
STOPWORDS_ID = set(stopwords.words("indonesian")).union(CUSTOM_STOPWORDS) - NEGASI_WORDS

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    STEMMER_AVAILABLE = True
    print("[INFO] PySastrawi loaded.")
except ImportError:
    STEMMER_AVAILABLE = False
    print("[WARNING] PySastrawi tidak ada. pip install PySastrawi")


# EMOJI MAP — emoji dikonversi ke kata sebelum cleaning
# supaya informasi sentimen dari emoji tidak hilang
EMOJI_MAP = {
    "👍": " bagus ", "👎": " buruk ", "😍": " suka ",  "🥰": " suka ",  "😘": " suka ",
    "❤️": " suka ",  "💕": " suka ",  "💞": " suka ", "😊": " senang ", "😁": " senang ",
    "🙏": " terima kasih ", "😭": " kecewa ", "😢": " kecewa ", "😡": " marah ",
    "🤢": " buruk ",  "🤮": " buruk ", "⭐": " bintang ", "🔥": " bagus ", "✨": " bagus ",
}

# KAMUS SKINCARE — kamus tambahan domain skincare
# melengkapi nasalsabila + new_kamusalay untuk istilah Tokopedia
KAMUS_SKINCARE = {
    "tak":"tidak", "gak":"tidak", "ga":"tidak", "ngga":"tidak", "nggak":"tidak",
    "tdk":"tidak", "gk":"tidak", "ngak":"tidak", "kagak":"tidak",
    "bgt":"banget", "bgtt":"banget", "bngt":"banget", "bangettt":"banget",
    "yg":"yang", "dg":"dengan", "dgn":"dengan", "utk":"untuk",
    "udh":"sudah", "udah":"sudah", "dah":"sudah",
    "blm":"belum", "belom":"belum", "jg":"juga", "jga":"juga",
    "krn":"karena", "karna":"karena", "klo":"kalau", "kalo":"kalau",
    "aja":"saja", "doang":"saja", "gue":"saya", "gw":"saya", "sy":"saya",
    "lo":"kamu", "lu":"kamu", "pake":"pakai", "pakek":"pakai",
    "dipake":"dipakai", "emang":"memang", "emg":"memang",
    "msh":"masih", "tetep":"tetap", "gabisa":"tidak bisa", "gbs":"tidak bisa",
    "sampe":"sampai", "nyampe":"sampai", "smua":"semua",
    "ngilangin":"menghilangkan", "ilangin":"menghilangkan", "ilang":"hilang",
    "ngurangin":"mengurangi", "ngatasin":"mengatasi",
    "nyembuhin":"menyembuhkan", "mudar":"memudar", "pudar":"memudar",
    "berkurang":"berkurang", "nyerap":"menyerap", "meresap":"meresap",
    "ngelembapkan":"melembapkan", "lembapkan":"melembapkan",
    "ngelebabpin":"melembapkan", "ngebantu":"membantu", "bantu":"membantu",
    "ngerasa":"merasa", "kerasa":"terasa", "berasa":"terasa",
    "bruntuaan":"berjerawat", "bruntusan":"berjerawat", "bruntus":"berjerawat",
    "jerawatan":"berjerawat", "gatel":"gatal", "lembab":"lembap",
    "minyakan":"berminyak", "glowing":"bercahaya", "pori2":"pori", "sensitip":"sensitif",
    "bestt":"bagus", "best":"bagus", "baguss":"bagus", "bagusss":"bagus",
    "cocokk":"cocok", "mantap":"bagus", "mantul":"bagus", "mantab":"bagus",
    "kece":"bagus", "oke":"baik", "ok":"baik", "jos":"bagus", "joss":"bagus",
    "jelek":"buruk", "ancur":"buruk",
    "rekomen":"rekomendasi", "recommended":"rekomendasi",
    "ampuh":"efektif", "manjur":"efektif",
    "worth":"sepadan", "worthit":"sepadan",
    "murah":"murah", "murce":"murah", "murmer":"murah",
    "muka":"wajah", "muke":"wajah", "jerawat":"jerawat", "jrawat":"jerawat",
    "iritasi":"iritasi", "skincare":"skincare", "skincr":"skincare",
    "moisturizer":"pelembap", "mois":"pelembap",
    "sunscreen":"tabir surya", "sesuwai":"sesuai",
    "ori":"original", "asli":"original", "palsu":"palsu", "kw":"palsu",
    "packaging":"kemasan", "packing":"kemasan",
    "makasih":"terima kasih", "tq":"terima kasih",
    "alhamdulillah":"bagus", "masyaallah":"bagus",
    "sih":"", "deh":"", "dong":"", "loh":"", "wkwk":"", "hehe":"", "haha":"",
}


def load_kamus_alay():
    if os.path.exists(KAMUS_PATH):
        print("[INFO] Memuat kamus dari cache...")
        with open(KAMUS_PATH, "r", encoding="utf-8") as f:
            kamus = json.load(f)
        if "tak" not in kamus:
            kamus["tak"] = "tidak"
        return kamus

    kamus = {}
    print("[INFO] Download kamus 1: nasalsabila/kamus-alay (Salsabila et al., 2018)...")
    try:
        r = requests.get(
            "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv",
            timeout=15
        )
        r.raise_for_status()
        for line in r.text.strip().split("\n")[1:]:
            p = line.split(",")
            if len(p) >= 2:
                s, f = p[0].strip().strip('"').lower(), p[1].strip().strip('"').lower()
                if s and f: kamus[s] = f
        print(f"  → {len(kamus)} entri")
    except Exception as e:
        print(f"  [WARNING] {e}")

    print("[INFO] Download kamus 2: new_kamusalay (Ibrohim & Budi, 2019)...")
    n = len(kamus)
    try:
        r = requests.get(
            "https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv",
            timeout=15
        )
        r.raise_for_status()
        for line in r.text.strip().split("\n")[1:]:
            p = line.split(",")
            if len(p) >= 2:
                s, f = p[0].strip().strip('"').lower(), p[1].strip().strip('"').lower()
                if s and f and s not in kamus: kamus[s] = f
        print(f"  → {len(kamus)-n} entri baru")
    except Exception as e:
        print(f"  [WARNING] {e}")

    n = len(kamus)
    kamus.update(KAMUS_SKINCARE)
    print(f"[INFO] Kamus skincare manual: {len(kamus)-n} entri baru")
    print(f"[INFO] Total: {len(kamus)} entri")

    with open(KAMUS_PATH, "w", encoding="utf-8") as f:
        json.dump(kamus, f, ensure_ascii=False, indent=2)
    return kamus

KAMUS_ALAY = load_kamus_alay()


# ── PREPROCESSING ──
# Urutan HARUS identik dengan predict.py supaya token training = token prediksi

def translate_emoji(text):
    for emoji, word in EMOJI_MAP.items():
        text = text.replace(emoji, word)
    return text

def clean_repeated(text):
    # "bagussss" → "baguss"
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def normalize_alay(text):
    return " ".join([v for w in text.split() for v in [KAMUS_ALAY.get(w, w)] if v])

def handle_negation(tokens):
    # "tidak" + "cocok" → "tidak_cocok" (satu token)
    # supaya stemmer tidak memisahkan maknanya
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] in NEGASI_WORDS and i + 1 < len(tokens):
            result.append(f"tidak_{tokens[i+1]}")
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

def is_noise(text):
    if not isinstance(text, str): return True
    t = text.strip()
    if len(t) < 5: return True
    ascii_only = t.encode("ascii", "ignore").decode("ascii").strip()
    alpha = re.sub(r'[^a-zA-Z]', '', ascii_only)
    if len(alpha) < 3:
        return False if len(t) >= 20 else True
    if len(set(alpha.lower())) / len(alpha) < 0.15: return True
    words = ascii_only.split()
    if words and any(len(w) > 25 for w in words): return True
    return False

def preprocess_text(text):
    if not isinstance(text, str) or len(text.strip()) < 3: return []
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


# ── LOAD DATASET ──
# Label dari rating: 1-2 → negatif, 4-5 → positif, 3 → dibuang (ambigu)
# Referensi: Liu (2012)

def load_sekarmulyani():
    if os.path.exists(TRAINING_CSV):
        print(f"[INFO] Dataset utama dari cache: {TRAINING_CSV}")
        df = pd.read_csv(TRAINING_CSV)
        if "source" not in df.columns: df["source"] = "sekarmulyani"
        return df

    print("[INFO] Download: sekarmulyani/ulasan-beauty-products")
    dataset = load_dataset("sekarmulyani/ulasan-beauty-products")
    frames  = [dataset[s].to_pandas() for s in ["train", "validation", "test"] if s in dataset]
    raw     = pd.concat(frames, ignore_index=True)

    rows = []
    for _, row in raw.iterrows():
        text = str(row.get("Review", "")).strip()
        if not text or len(text) < 5: continue
        if   row.get("Bintang 1") or row.get("Bintang 2"): label = "negative"
        elif row.get("Bintang 4") or row.get("Bintang 5"): label = "positive"
        else: continue
        rows.append({"review_text": text, "label": label, "source": "sekarmulyani"})

    df = pd.DataFrame(rows).drop_duplicates(subset=["review_text"])
    df.to_csv(TRAINING_CSV, index=False)
    print(f"[INFO] Dataset utama: {len(df)} baris")
    return df

def load_csv_domain(csv_path, source_name):
    if not os.path.exists(csv_path):
        print(f"[WARNING] {source_name} tidak ditemukan: {csv_path}")
        return pd.DataFrame(columns=["review_text", "label", "source"])

    print(f"[INFO] Domain adaptation: {csv_path}")
    df = None
    for sep in [",", ";"]:
        try:
            tmp = pd.read_csv(csv_path, sep=sep, encoding="utf-8-sig")
            if len(tmp.columns) > 1: df = tmp; break
        except: continue

    if df is None:
        return pd.DataFrame(columns=["review_text", "label", "source"])

    rows = []
    for _, row in df.iterrows():
        text   = str(row.get("Comment", "")).strip()
        rating = row.get("Rating", 0)
        if not text or len(text) < 5: continue
        try: rating = float(rating)
        except: continue
        if   rating <= 2: label = "negative"
        elif rating >= 4: label = "positive"
        else: continue  # rating 3 dibuang — labelnya ambigu (Liu, 2012)
        rows.append({"review_text": text, "label": label, "source": source_name})

    result = pd.DataFrame(rows).drop_duplicates(subset=["review_text"])
    print(f"[INFO] {source_name}: {len(result)} baris | {result['label'].value_counts().to_dict()}")
    return result

def load_g2g():   return load_csv_domain(G2G_CSV,   "g2g_tokopedia")
def load_heyxi(): return load_csv_domain(HEYXI_CSV, "heyxi_tokopedia")


def build_bigram(corpus):
    print("[INFO] Bigram model...")
    phraser = Phraser(Phrases(corpus, min_count=5, threshold=10, delimiter="_"))
    joblib.dump(phraser, BIGRAM_PATH)
    return phraser

def train_w2v(corpus):
    # Word2Vec Skip-gram (sg=1): lebih baik untuk kata jarang (Mikolov et al., 2013)
    # epochs=20: vektor lebih stabil, seed=42: reproducible
    print("[INFO] Word2Vec training (epochs=20)...")
    model = Word2Vec(
        sentences=corpus, vector_size=100, window=5,
        min_count=2, sg=1, workers=4, epochs=20, seed=42
    )
    model.save(os.path.join(MODEL_DIR, "word2vec.model"))
    return model

def doc_vector(tokens, model):
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)


def main():
    print("=" * 60)
    print("  TRAINING — Skincare Sentiment Analysis")
    print("  Kamus  : nasalsabila + new_kamusalay + skincare manual")
    print("  Dataset: sekarmulyani + G2G Tokopedia + Heyxi Tokopedia")
    print("=" * 60)

    # STEP 1 — gabungkan semua dataset
    print("\n[STEP 1] Load dataset...")
    df = pd.concat([load_sekarmulyani(), load_g2g(), load_heyxi()], ignore_index=True)
    df = df.dropna(subset=["review_text", "label"])
    df = df[df["label"].isin(["positive", "negative"])]
    df = df.drop_duplicates(subset=["review_text"])

    before = len(df)
    df = df[~df["review_text"].apply(is_noise)]
    print(f"[INFO] Noise dibuang: {before - len(df)} baris")

    n_sek   = len(df[df["source"] == "sekarmulyani"])
    n_g2g   = len(df[df["source"] == "g2g_tokopedia"])
    n_heyxi = len(df[df["source"] == "heyxi_tokopedia"])

    print(f"       Total Data Bersih : {len(df)}")
    print(f"       - Sekarmulyani  : {n_sek}")
    print(f"       - G2G Tokopedia : {n_g2g}")
    print(f"       - Heyxi         : {n_heyxi}")

    print("\n Rincian Distribusi (Positif vs Negatif):")
    print(" 1. KESELURUHAN")
    print(df['label'].value_counts().to_string())
    
    print("\n 2. SEKARMULYANI")
    print(df[df['source'] == 'sekarmulyani']['label'].value_counts().to_string())
    
    print("\n 3. G2G TOKOPEDIA")
    print(df[df['source'] == 'g2g_tokopedia']['label'].value_counts().to_string())
    
    print("\n 4. HEYXI TOKOPEDIA")
    print(df[df['source'] == 'heyxi_tokopedia']['label'].value_counts().to_string())

    # STEP 2 — preprocessing
    print("\n[STEP 2] Preprocessing...")
    df["tokens"] = df["review_text"].apply(preprocess_text)
    df = df[df["tokens"].map(len) > 0]
    neg_samples = [t for tokens in df["tokens"][:500] for t in tokens if t.startswith("tidak_")]
    print(f"[INFO] Setelah preprocessing: {len(df)} baris")
    print(f"[INFO] Contoh token negasi: {[t for t, _ in Counter(neg_samples).most_common(8)]}")

    # STEP 3 — bigram
    print("\n[STEP 3] Bigram detection...")
    phraser      = build_bigram(df["tokens"].tolist())
    df["tokens"] = df["tokens"].apply(lambda t: list(phraser[t]))
    bg_samples   = [t for tokens in df["tokens"][:500] for t in tokens if "_" in t and not t.startswith("tidak_")]
    print(f"[INFO] Contoh bigram: {[b for b, _ in Counter(bg_samples).most_common(8)]}")

    # STEP 4 — Word2Vec
    print("\n[STEP 4] Word2Vec training...")
    w2v = train_w2v(df["tokens"].tolist())
    joblib.dump(w2v, os.path.join(MODEL_DIR, "vectorizer.pkl"))

    # STEP 5 — document vectors
    print("\n[STEP 5] Document vectors...")
    X = np.array([doc_vector(t, w2v) for t in df["tokens"]])
    y = df["label"].values
    print(f"[INFO] Shape: {X.shape}")

    # STEP 6 — split 80/20
    # stratify=y: proporsi positif/negatif di train dan test tetap sama
    print("\n[STEP 6] Split 80/20...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_tr)} | Test: {len(X_te)}")

    # STEP 7 — GridSearchCV
    # f1_macro dipilih karena data imbalanced (lebih banyak positif)
    # class_weight balanced: kelas minoritas (negatif) diberi bobot lebih
    print("\n[STEP 7] GridSearchCV — hyperparameter tuning...")
    print("[INFO] Parameter: C=[0.1, 1.0, 10.0], class_weight=[None, balanced]")
    print("[INFO] Scoring: f1_macro, CV: 5-fold")

    param_grid = {
        "C":            [0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
        param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1
    )
    grid.fit(X_tr, y_tr)

    print(f"\n[INFO] Parameter terbaik: {grid.best_params_}")
    print(f"[INFO] F1-macro rata-rata 5-fold: {grid.best_score_:.4f}")

    lr = grid.best_estimator_
    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

    # STEP 8 — evaluasi
    print("\n[STEP 8] Evaluasi pada data test...")
    y_pred = lr.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    pre = precision_score(y_te, y_pred, average="weighted")
    rec = recall_score(y_te,    y_pred, average="weighted")
    f1  = f1_score(y_te,        y_pred, average="weighted")

    classes        = lr.classes_.tolist()
    cm             = confusion_matrix(y_te, y_pred, labels=classes)
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    print(f"\n  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {pre:.4f} ({pre*100:.2f}%)")
    print(f"  Recall    : {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
    print(f"\n  Confusion Matrix:")
    print(f"                 | Pred Negatif | Pred Positif")
    print(f"  Aktual Negatif | {tn:12} | {fp:12}")
    print(f"  Aktual Positif | {fn:12} | {tp:12}")
    print(classification_report(y_te, y_pred))

    # STEP 9 — simpan model_performance.json
    grid_results = []
    for params, mean_score, std_score in zip(
        grid.cv_results_["params"],
        grid.cv_results_["mean_test_score"],
        grid.cv_results_["std_test_score"]
    ):
        grid_results.append({
            "params":  str(params),
            "mean_f1": round(float(mean_score), 4),
            "std_f1":  round(float(std_score),  4),
        })

    perf = {
        "accuracy":      round(acc * 100, 2),
        "precision":     round(pre * 100, 2),
        "recall":        round(rec * 100, 2),
        "f1_score":      round(f1  * 100, 2),
        "training_size": int(len(X_tr)),
        "testing_size":  int(len(X_te)),
        "total_data":    int(len(df)),
        "confusion_matrix": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "labels": classes,
        },
        "best_params":         grid.best_params_,
        "grid_search_results": grid_results,
        "datasets": [
            f"sekarmulyani: {n_sek}",
            f"G2G Tokopedia: {n_g2g}",
            f"Heyxi Tokopedia: {n_heyxi}",
        ],
        "kamus_normalisasi": [
            "nasalsabila/kamus-alay (Salsabila et al., 2018)",
            "okkyibrohim/new_kamusalay (Ibrohim & Budi, 2019)",
            "Kamus skincare tambahan (dikurasi manual)",
        ],
        "algorithm":    "Word2Vec Skip-gram (epochs=20) + Bigram + Logistic Regression (GridSearchCV)",
        "clustering":   "Agglomerative Hierarchical Clustering (average linkage) + Silhouette Score",
        "preprocessing": [
            "Terjemahan Emoji",
            "Noise Filter",
            "Normalisasi Huruf Berulang",
            "Cleaning (URL, mention, hashtag, unicode)",
            "Case Folding",
            "Normalisasi Alay (3 kamus)",
            "Tokenizing",
            "Stopword Removal (kata negasi dipertahankan)",
            "Negasi Handling",
            "Stemming PySastrawi",
            "Bigram Detection",
        ],
        "label_method":          "Rating 1-2=Negatif, 4-5=Positif, 3=Dihapus (Liu, 2012)",
        "hyperparameter_tuning": "GridSearchCV 5-fold CV, scoring=f1_macro",
    }

    with open(os.path.join(MODEL_DIR, "model_performance.json"), "w", encoding="utf-8") as f:
        json.dump(perf, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"  TRAINING SELESAI")
    print(f"  Total data : {len(df)}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  F1-Score   : {f1*100:.2f}%")
    print(f"  Best params: {grid.best_params_}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print("=" * 60)


if __name__ == "__main__":
    main()