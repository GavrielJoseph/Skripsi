import os, re, sys, json, joblib, warnings
import pandas as pd
import numpy as np
import nltk

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import Counter

warnings.filterwarnings("ignore")


BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data", "raw")

LR_MODEL_PATH  = os.path.join(MODEL_DIR, "logistic_regression.pkl")
W2V_MODEL_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
BIGRAM_PATH    = os.path.join(MODEL_DIR, "bigram_phraser.pkl")
KAMUS_PATH     = os.path.join(DATA_DIR,  "kamus_alay.json")

nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

NEGASI_WORDS = {"tidak", "bukan", "jangan", "belum", "tanpa", "kurang"}
STOPWORDS_ID = set(stopwords.words("indonesian")) - NEGASI_WORDS
# kata negasi sengaja TIDAK dimasukkan stopwords
# karena mengubah makna sentimen ("tidak cocok" != "cocok")

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    STEMMER_AVAILABLE = True
except ImportError:
    STEMMER_AVAILABLE = False


# EMOJI MAP untuk emoji yang sering muncul pada ulasan tokopedia
EMOJI_MAP = {
    "👍": " bagus ", "👎": " buruk ",
    "😍": " suka ",  "🥰": " suka ",  "😘": " suka ",
    "❤️": " suka ",  "💕": " suka ",  "💞": " suka ",
    "😊": " senang ", "😁": " senang ", "🙏": " terima kasih ",
    "😭": " kecewa ", "😢": " kecewa ", "😡": " marah ",
    "🤢": " buruk ",  "🤮": " buruk ",
    "⭐": " bintang ", "🔥": " bagus ", "✨": " bagus ",
}


TOPIC_KEYWORDS = {
    "efek_produk": [
        "jerawat","bruntusan","cerah","lembap","lembab","kering","berminyak","iritasi",
        "sensitif","kulit","wajah","muka","glowing","bercahaya","mulus","halus","pori",
        "bekas","merah","kemerahan","barrier","skin","tekstur","formula","ceramide",
        "hyaluronic","niacinamide","kandungan","bahan","serum","moisturizer","pelembap",
        "sunscreen","cocok","alergi","perih","gatal","meradang","flek","noda"
    ],
    "pengiriman": [
        "kirim","kurir","paket","ekspedisi","pengiriman","tiba",
        "ongkir","ongkos","estimasi","tracking","resi",
        "jnt","jne","sicepat","anteraja","gosend",
        "bocor","segel",
    ],
    "harga_nilai": [
        "harga","mahal","murah","worth","sepadan","promo","diskon","hemat","terjangkau",
        "kualitas","harganya","budget","ekonomis","cashback","voucher","flash","sale"
    ],
    "kemasan": [
        "kemasan","packaging","packing","botol","tube","pot","pump","wadah","kardus",
        "bubble","wrap","pecah","rapih","rapi","cantik","desain"
    ],
    "pembelian": [
        "beli","order","pesan","langganan","ulang","repeat","restock","kali","refill",
        "official","toko","seller","respon","reseller","ori","original","palsu","asli"
    ],
}

TOPIC_WEIGHT    = 3.0
ALL_TOPIC_WORDS = set(w for words in TOPIC_KEYWORDS.values() for w in words)

# Label tampilan per topik untuk UI
TOPIC_LABELS = {
    "efek_produk": "Efek Produk",
    "pengiriman":  "Pengiriman",
    "harga_nilai": "Harga & Nilai",
    "kemasan":     "Kemasan",
    "pembelian":   "Pembelian & Keaslian",
}


# KAMUS ALAY
# File ini berisi gabungan 3 sumber:
# 1. nasalsabila/kamus-alay (Salsabila et al., 2018)
# 2. okkyibrohim/new_kamusalay (Ibrohim & Budi, 2019)
# 3. Kamus skincare manual (dikurasi sendiri dan bantuan AI)
KAMUS_FALLBACK = {
    "tak":"tidak","gak":"tidak","ga":"tidak","ngga":"tidak","nggak":"tidak",
    "tdk":"tidak","gk":"tidak","ngak":"tidak","kagak":"tidak",
    "bgt":"banget","yg":"yang","dg":"dengan","utk":"untuk",
    "udh":"sudah","udah":"sudah","blm":"belum","jg":"juga",
    "krn":"karena","klo":"kalau","kalo":"kalau","aja":"saja",
    "gue":"saya","gw":"saya","lo":"kamu","pake":"pakai",
    "emang":"memang","msh":"masih","sampe":"sampai",
    "bruntusan":"berjerawat","gatel":"gatal","lembab":"lembap",
    "glowing":"bercahaya","bestt":"bagus","best":"bagus",
    "mantap":"bagus","oke":"baik","ok":"baik","jelek":"buruk",
    "rekomen":"rekomendasi","ampuh":"efektif","worth":"sepadan",
    "murah":"murah","muka":"wajah","iritasi":"iritasi",
    "skincare":"skincare","moisturizer":"pelembap","sunscreen":"tabir surya",
    "ori":"original","packaging":"kemasan","makasih":"terima kasih",
    "alhamdulillah":"bagus","sih":"","deh":"","dong":"","wkwk":"",
}

def load_kamus():
    if os.path.exists(KAMUS_PATH):
        with open(KAMUS_PATH, "r", encoding="utf-8") as f:
            kamus = json.load(f)
        if "tak" not in kamus:
            kamus["tak"] = "tidak"
        return kamus
    return KAMUS_FALLBACK

KAMUS_ALAY = load_kamus()


def translate_emoji(text):
    for emoji, word in EMOJI_MAP.items():
        text = text.replace(emoji, word)
    return text

def clean_repeated_chars(text):
    # regex (.)\1{2,} cocok dengan karakter apapun yang muncul > 2x berturut
    # \1 adalah backreference ke karakter yang match
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def normalize_alay(text):
    return " ".join([v for w in text.split() for v in [KAMUS_ALAY.get(w, w)] if v])

def handle_negation(tokens):
    # Gabungkan kata negasi dengan kata berikutnya
    # "tidak cocok" → "tidak_cocok" (satu token)
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

def preprocess_text(text):
    if not isinstance(text, str) or len(text.strip()) < 3:
        return []
    text = translate_emoji(text)
    text = clean_repeated_chars(text)
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = normalize_alay(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) >= 2]
    tokens = handle_negation(tokens)
    if STEMMER_AVAILABLE:
        # token tidak_ dikecualikan dari stemming
        # karena PySastrawi akan memecah "tidak_cocok" menjadi "tidak" + "cocok"
        tokens = [t if t.startswith("tidak_") else stemmer.stem(t) for t in tokens]
    return tokens

def apply_bigram(tokens, phraser):
    if phraser is None:
        return tokens
    return list(phraser[tokens])

def is_noise(text):
    # Filter teks yang tidak punya konten linguistik yang bisa dianalisis.
    # Tiga layer check — dari yang paling cepat ke paling teliti.
    if not isinstance(text, str):
        return True
    t = text.strip()
    if len(t) < 5:
        return True

    ascii_only = t.encode("ascii", "ignore").decode("ascii").strip()
    alpha = re.sub(r'[^a-zA-Z]', '', ascii_only)

    if len(alpha) < 3:
        if len(t) >= 20:
            return False
        return True

    words = ascii_only.lower().split()
    if len(words) >= 10:
        if len(set(words)) / len(words) >= 0.4:
            return False

    ratio = len(set(alpha.lower())) / len(alpha)
    if len(alpha) > 200:
        threshold = 0.06
    elif len(alpha) > 100:
        threshold = 0.08
    elif len(alpha) > 50:
        threshold = 0.10
    else:
        threshold = 0.15

    if ratio < threshold:
        return True

    tokens = [tok for tok in re.split(r'[^a-zA-Z]+', ascii_only) if tok]
    if any(len(tok) > 25 for tok in tokens):
        return True

    return False


# WORD2VEC VECTORS
def get_document_vector(tokens, w2v_model):
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)

def get_topic_weighted_vector(tokens, w2v_model):
    vectors = []
    weights = []
    for t in tokens:
        raw = t.replace("tidak_", "")
        if t in w2v_model.wv:
            w = TOPIC_WEIGHT if raw in ALL_TOPIC_WORDS else 1.0
            vectors.append(w2v_model.wv[t])
            weights.append(w)
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    return np.average(np.array(vectors), axis=0, weights=np.array(weights))


# LOAD MODEL
def load_models():
    for path in [LR_MODEL_PATH, W2V_MODEL_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    lr_model  = joblib.load(LR_MODEL_PATH)
    w2v_model = joblib.load(W2V_MODEL_PATH)
    phraser   = joblib.load(BIGRAM_PATH) if os.path.exists(BIGRAM_PATH) else None
    return lr_model, w2v_model, phraser


# LOAD CSV
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
    for c in ["comment", "review_text", "review", "ulasan", "text", "komentar"]:
        if c in col_map:
            return col_map[c]
    return df.columns[0]

def find_product_column(df):
    col_map = {c.lower(): c for c in df.columns}
    for c in ["productname", "product_name", "product", "produk", "nama_produk"]:
        if c in col_map:
            return col_map[c]
    return None


# TOPIC DETECTION DARI TEKS MENTAH
# Berbeda dengan TOPIC_KEYWORDS yang dipakai untuk Word2Vec weighting,
# fungsi ini bekerja pada teks mentah sebelum preprocessing sehingga
# kata informal seperti "packing", "ongkir", "ori" tetap terdeteksi.
# Hasilnya dipakai untuk topic_breakdown dan topic_overview di output.
TOPIC_RAW_KEYWORDS = {
    "efek_produk": [
        "jerawat","bruntusan","iritasi","gatal","perih","alergi","kulit","wajah","muka",
        "lembap","lembab","kering","cerah","glowing","cocok","formula","bahan","serum",
        "bekas","flek","noda","pori","halus","mulus","berminyak","sensitif","breakout",
        "merah","kemerahan","meradang","moisturizer","pelembap","sunscreen","niacinamide",
        "ceramide","hyaluronic","tekstur","barrier","kandungan"
    ],
    "pengiriman": [
        "kirim","sampai","kurir","paket","ekspedisi","pengiriman","lambat","lama",
        "ongkir","ongkos","resi","jnt","jne","sicepat","anteraja","gosend","tiba",
        "datang","estimasi","tracking","cepat"
    ],
    "harga_nilai": [
        "harga","mahal","murah","worth","sepadan","promo","diskon","hemat","terjangkau",
        "budget","ekonomis","cashback","voucher","sale","harganya","worth it"
    ],
    "kemasan": [
        "kemasan","packing","packaging","botol","tube","pot","pump","wadah",
        "bocor","pecah","rusak","segel","kardus","bubble","wrap","rapi","rapih","kemas"
    ],
    "pembelian": [
        "palsu","kw","ori","original","asli","seller","toko","reseller","official",
        "repeat","ulang","restock","langganan","respon","beli lagi","kedua kali"
    ],
}

def detect_topic_from_raw(text):
    """Deteksi aspek topik dari teks mentah. Return list topik yang terdeteksi."""
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    found = []
    for topic, words in TOPIC_RAW_KEYWORDS.items():
        if any(w in text_lower for w in words):
            found.append(topic)
    return found


# CLUSTERING — AHC (Agglomerative Hierarchical Clustering)
def find_optimal_k(X, max_k):
    n = len(X)

    # batas bawah: kalau data < 20, langsung return k=2
    # karena Silhouette tidak stabil di data sangat kecil
    if n < 20:
        return 2, -1.0

    best_k     = 2
    best_score = -1.0

    # range k: dari 2 sampai min(max_k, n//10+1)
    # batas n//10 mencegah k terlalu dekat dengan jumlah data
    # (AHC tidak stabil kalau k > 10% dari n)
    for k in range(2, min(max_k + 1, n // 10 + 2)):
        try:
            labels = AgglomerativeClustering(
                n_clusters=k,
                linkage="average"  # average linkage = UPGMA
            ).fit_predict(X)

            if len(set(labels)) < 2:
                continue

            # sample_size=500 untuk efisiensi:
            # menghitung Silhouette butuh O(n^2) jarak
            score = silhouette_score(
                X, labels,
                sample_size=min(500, n),
                random_state=42
            )

            if score > best_score:
                best_score = score
                best_k     = k

        except Exception:
            continue

    return best_k, round(float(best_score), 4)


def merge_small_clusters(cluster_labels, X_cluster, n_clusters, min_size):
    # cluster < min_size digabung ke cluster besar terdekat berdasarkan jarak centroid
    labels    = cluster_labels.copy()
    centroids = {}

    for cid in range(n_clusters):
        mask = labels == cid
        if mask.sum() > 0:
            centroids[cid] = X_cluster[mask].mean(axis=0)

    big_clusters = [
        cid for cid in range(n_clusters)
        if np.sum(labels == cid) >= min_size
    ]

    # FIX: kondisi asli "< 2" menghentikan merge saat hanya ada 1 big cluster,
    # padahal cluster kecil tetap bisa di-merge ke 1 big cluster yang ada.
    # Kondisi yang benar: stop hanya jika tidak ada big cluster sama sekali.
    # Contoh kasus yang terfix: 1221 vs 2 vs 1 vs 1 ulasan per cluster →
    # sebelumnya merge berhenti karena hanya 1 big cluster,
    # sekarang cluster 2-ulasan dan 1-ulasan di-merge ke cluster 1221-ulasan.
    if len(big_clusters) < 1:
        return labels

    for cid in range(n_clusters):
        if cid in big_clusters:
            continue
        if np.sum(labels == cid) == 0:
            continue

        dists = {
            big: np.linalg.norm(centroids[cid] - centroids[big])
            for big in big_clusters
            if big in centroids
        }
        if not dists:
            continue

        nearest_big = min(dists, key=dists.get)
        labels[labels == cid] = nearest_big

    unique_ids = sorted(set(labels))
    remap      = {old: new for new, old in enumerate(unique_ids)}
    return np.array([remap[c] for c in labels])


# KEYWORD EXTRACTION — Relative Frequency Ratio
def get_distinctive_keywords(cluster_id, all_tokens, cluster_labels, top_n=10):
    cluster_tokens = []
    all_corpus     = []

    for tokens, label in zip(all_tokens, cluster_labels):
        # token tidak_ dikeluarkan dari keyword karena bentuknya tidak natural
        # untuk ditampilkan ke user ("tidak_cocok")
        clean = [t for t in tokens if not t.startswith("tidak_")]
        all_corpus.extend(clean)
        if label == cluster_id:
            cluster_tokens.extend(clean)

    if not cluster_tokens:
        return []

    cluster_freq  = Counter(cluster_tokens)
    cluster_total = max(len(cluster_tokens), 1)
    corpus_freq   = Counter(all_corpus)
    corpus_total  = max(len(all_corpus), 1)

    scores = {}
    for word, count in cluster_freq.items():
        if len(word) < 3 or count < 2:
            continue
        ratio = (count / cluster_total) / max(corpus_freq[word] / corpus_total, 0.0001)
        scores[word] = ratio

    return [w for w, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]


# GENERATE CLUSTER NAME
# Prioritas 1 : kata yang terdaftar di TOPIC_KEYWORDS (domain skincare eksplisit)
# Prioritas 2 : kata Indonesia murni panjang >= 4, tanpa underscore
# Prioritas 3 : fallback apapun selain token negasi
# Token bigram non-topik (moga_cocok, fast_respon) tidak ditampilkan
# karena merupakan artefak teknis pipeline, bukan label bermakna
def generate_cluster_name(keywords):
    if not keywords:
        return "Cluster Umum"

    clean_keywords = [
        k for k in keywords
        if not k.startswith("tidak_")
        and not ("_" in k and k.replace("_", "") not in ALL_TOPIC_WORDS)
    ]

    if not clean_keywords:
        clean_keywords = [k for k in keywords if not k.startswith("tidak_")]
    if not clean_keywords:
        return "Cluster Umum"

    topic_words = [
        k for k in clean_keywords
        if k in ALL_TOPIC_WORDS and len(k) >= 4
    ]
    if topic_words:
        return " & ".join(w.capitalize() for w in topic_words[:2])

    indo_words = [k for k in clean_keywords if "_" not in k and len(k) >= 4]
    if indo_words:
        return " & ".join(w.capitalize() for w in indo_words[:2])

    return " & ".join(w.capitalize() for w in clean_keywords[:2])


# TOPIC BREAKDOWN PER CLUSTER
# Menghitung distribusi aspek topik bisnis dalam setiap cluster.
# Menjawab: "Dalam cluster ini, ulasan tentang efek produk
# bersentimen positif atau negatif?"
def compute_topic_breakdown(cluster_df):
    breakdown = {}
    for topic_key, label in TOPIC_LABELS.items():
        mask   = cluster_df["detected_topics"].apply(lambda t: topic_key in t)
        subset = cluster_df[mask]
        if len(subset) == 0:
            continue
        pos = int((subset["sentiment"] == "positive").sum())
        neg = int((subset["sentiment"] == "negative").sum())
        breakdown[topic_key] = {
            "label":    label,
            "count":    int(len(subset)),
            "positive": pos,
            "negative": neg,
            "pos_pct":  round(pos / len(subset) * 100, 1) if len(subset) > 0 else 0.0,
        }
    return breakdown


# SAMPLE REVIEWS PER CLUSTER
# Mengambil contoh ulasan representatif berdasarkan confidence tertinggi.
# Kriteria:
#   Positif : panjang >= 30 karakter (ulasan informatif, bukan sekadar "bagus")
#   Negatif : panjang >= 10 karakter
# Maksimal 3 sampel per sentimen.
def get_sample_reviews(cluster_df, n=3):
    samples = {"positive": [], "negative": []}
    for sentiment, min_len in [("positive", 30), ("negative", 10)]:
        subset = cluster_df[
            (cluster_df["sentiment"] == sentiment) &
            (cluster_df["review_text"].str.len() >= min_len)
        ].sort_values("confidence", ascending=False).head(n)
        samples[sentiment] = subset["review_text"].tolist()
    return samples


# ============================================================
# MAIN — alur eksekusi utama
# ============================================================
def main():
    if len(sys.argv) < 2:
        sys.stdout.buffer.write(json.dumps(
            {"status": "error", "message": "Tidak ada file CSV."},
            ensure_ascii=False
        ).encode("utf-8"))
        sys.exit(1)

    csv_path       = sys.argv[1]
    ui_text_col    = sys.argv[2] if len(sys.argv) > 2 else ""
    ui_product_col = sys.argv[3] if len(sys.argv) > 3 else ""

    if not os.path.exists(csv_path):
        sys.stdout.buffer.write(json.dumps(
            {"status": "error", "message": f"File tidak ditemukan: {csv_path}"},
            ensure_ascii=False
        ).encode("utf-8"))
        sys.exit(1)

    try:
        df = load_csv(csv_path)
    except Exception as e:
        sys.stdout.buffer.write(json.dumps(
            {"status": "error", "message": f"Gagal membaca CSV: {e}"},
            ensure_ascii=False
        ).encode("utf-8"))
        sys.exit(1)

    text_col    = ui_text_col    if ui_text_col    and ui_text_col    in df.columns else find_text_column(df)
    product_col = ui_product_col if ui_product_col and ui_product_col in df.columns else find_product_column(df)

    if text_col not in df.columns:
        sys.stdout.buffer.write(json.dumps(
            {"status": "error", "message": f"Kolom '{text_col}' tidak ada di CSV."},
            ensure_ascii=False
        ).encode("utf-8"))
        sys.exit(1)

    df["review_text"]  = df[text_col].astype(str)
    df["product_name"] = df[product_col].astype(str) if product_col else ""

    df          = df.dropna(subset=["review_text"]).reset_index(drop=True)
    total_input = len(df)

    if total_input == 0:
        sys.stdout.buffer.write(json.dumps(
            {"status": "error", "message": "Tidak ada data ulasan yang valid."},
            ensure_ascii=False
        ).encode("utf-8"))
        sys.exit(1)

    try:
        lr_model, w2v_model, phraser = load_models()
    except FileNotFoundError as e:
        sys.stdout.buffer.write(json.dumps(
            {"status": "error", "message": str(e)},
            ensure_ascii=False
        ).encode("utf-8"))
        sys.exit(1)

    # deteksi topik dari teks mentah SEBELUM preprocessing
    # agar kata informal seperti "packing", "ongkir" masih bisa dideteksi
    df["detected_topics"] = df["review_text"].apply(detect_topic_from_raw)

    # preprocessing untuk ML
    df["tokens"] = df["review_text"].apply(preprocess_text)
    df["tokens"] = df["tokens"].apply(lambda t: apply_bigram(t, phraser))

    skipped_mask  = df["tokens"].apply(len) == 0
    skipped_count = int(skipped_mask.sum())
    df            = df[~skipped_mask].reset_index(drop=True)

    if len(df) == 0:
        sys.stdout.buffer.write(json.dumps(
            {"status": "error", "message": "Semua ulasan kosong setelah preprocessing."},
            ensure_ascii=False
        ).encode("utf-8"))
        sys.exit(1)

    X_sentiment = np.array([get_document_vector(t, w2v_model)       for t in df["tokens"]])
    X_cluster   = np.array([get_topic_weighted_vector(t, w2v_model) for t in df["tokens"]])

    # KLASIFIKASI SENTIMEN (Logistic Regression)
    # predict() → kelas dengan probabilitas tertinggi ("positive" atau "negative")
    # predict_proba() → probabilitas untuk setiap kelas [P(negatif), P(positif)]
    df["sentiment"] = lr_model.predict(X_sentiment)
    proba           = lr_model.predict_proba(X_sentiment)
    classes         = list(lr_model.classes_)

    df["confidence"] = [
        round(float(proba[i][classes.index(df["sentiment"][i])]), 4)
        for i in range(len(df))
    ]

    conf_values = df["confidence"].tolist()
    confidence_distribution = {
        "high":   int(sum(1 for c in conf_values if c >= 0.8)),
        "medium": int(sum(1 for c in conf_values if 0.6 <= c < 0.8)),
        "low":    int(sum(1 for c in conf_values if c < 0.6)),
    }

    # CLUSTERING (AHC)
    # hanya ulasan dengan >= 3 token yang masuk clustering
    valid_mask = df["tokens"].apply(len) >= 3
    X_valid    = X_cluster[valid_mask]
    idx_valid  = df[valid_mask].index.tolist()
    n          = len(X_valid)

    # max_k = min(8, max(3, n // 200))
    # untuk n=1225: max_k = min(8, max(3, 6)) = 6
    # untuk n=649:  max_k = min(8, max(3, 3)) = 3
    max_k = min(8, max(3, n // 200))

    n_clusters, best_silhouette = find_optimal_k(X_valid, max_k)

    # jalankan AHC dengan k yang terpilih
    # linkage="average" = UPGMA (Unweighted Pair Group Method with Arithmetic Mean)
    labels_valid = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="average"
    ).fit_predict(X_valid)

    centroids_init = np.array([
        X_valid[labels_valid == cid].mean(axis=0)
        if (labels_valid == cid).any()
        else np.zeros(X_valid.shape[1])
        for cid in range(n_clusters)
    ])

    cluster_labels = np.full(len(df), -1, dtype=int)
    for i, idx in enumerate(idx_valid):
        cluster_labels[idx] = labels_valid[i]

    for i in range(len(df)):
        if cluster_labels[i] == -1:
            cluster_labels[i] = int(
                np.argmin(np.linalg.norm(centroids_init - X_cluster[i], axis=1))
            )

    # min_cluster_size = max(3, 0.5% dari total data)
    # 0.5% dipilih karena lebih konservatif dari 0.2% sebelumnya:
    # untuk n=1225: max(3, int(1225*0.005)) = max(3, 6) = 6
    # untuk n=649:  max(3, int(649*0.005))  = max(3, 3) = 3
    # Ini memastikan cluster dengan < 6 ulasan (untuk dataset ~1225)
    # selalu di-merge ke cluster terdekat yang cukup besar
    min_cluster_size = max(3, int(len(df) * 0.005))

    needs_merge = any(
        np.sum(cluster_labels == cid) < min_cluster_size
        for cid in range(n_clusters)
    )
    if needs_merge:
        cluster_labels = merge_small_clusters(
            cluster_labels, X_cluster, n_clusters, min_cluster_size
        )

    n_clusters_final = len(set(cluster_labels))
    if n_clusters_final < 2:
        labels_valid_2 = AgglomerativeClustering(
            n_clusters=2, linkage="average"
        ).fit_predict(X_valid)

        cluster_labels = np.full(len(df), -1, dtype=int)
        for i, idx in enumerate(idx_valid):
            cluster_labels[idx] = labels_valid_2[i]

        c2 = np.array([
            X_valid[labels_valid_2 == cid].mean(axis=0)
            if (labels_valid_2 == cid).any()
            else np.zeros(X_valid.shape[1])
            for cid in range(2)
        ])
        for i in range(len(df)):
            if cluster_labels[i] == -1:
                cluster_labels[i] = int(
                    np.argmin(np.linalg.norm(c2 - X_cluster[i], axis=1))
                )

        n_clusters_final = 2

    df["cluster_id"] = cluster_labels.tolist()

    final_silhouette = best_silhouette
    if n_clusters_final >= 2 and len(df) >= 10:
        try:
            final_silhouette = round(float(silhouette_score(
                X_cluster, cluster_labels,
                sample_size=min(500, len(df)),
                random_state=42
            )), 4)
        except Exception:
            pass

    all_tokens   = df["tokens"].tolist()
    cluster_info = {}

    for cid in range(n_clusters_final):
        mask       = cluster_labels == cid
        cluster_df = df[mask].copy()
        count      = int(mask.sum())
        keywords   = get_distinctive_keywords(cid, all_tokens, cluster_labels, top_n=10)
        name       = generate_cluster_name(keywords)

        cluster_info[str(cid)] = {
            "label":           name,
            "count":           count,
            "positive":        int((cluster_df["sentiment"] == "positive").sum()),
            "negative":        int((cluster_df["sentiment"] == "negative").sum()),
            "avg_confidence":  round(float(cluster_df["confidence"].mean()), 4),
            "top_keywords":    keywords,
            "is_small":        count < min_cluster_size,
            "topic_breakdown": compute_topic_breakdown(cluster_df),
            "sample_reviews":  get_sample_reviews(cluster_df, n=3),
        }

    # topic_overview: distribusi aspek topik di seluruh dataset
    # Menjawab: aspek mana yang paling banyak dibahas dan bagaimana sentimennya?
    topic_overview = {}
    for topic_key, label in TOPIC_LABELS.items():
        mask   = df["detected_topics"].apply(lambda t: topic_key in t)
        subset = df[mask]
        if len(subset) == 0:
            continue
        pos = int((subset["sentiment"] == "positive").sum())
        neg = int((subset["sentiment"] == "negative").sum())
        topic_overview[topic_key] = {
            "label":    label,
            "count":    int(len(subset)),
            "positive": pos,
            "negative": neg,
            "pos_pct":  round(pos / len(subset) * 100, 1) if len(subset) > 0 else 0.0,
        }

    df["keywords"] = df["tokens"].apply(
        lambda t: [w for w, _ in Counter(
            [x for x in t if not x.startswith("tidak_")]
        ).most_common(5)]
    )

    sentiment_counts = df["sentiment"].value_counts().to_dict()

    results = [
        {
            "product_name": row["product_name"],
            "review_text":  row["review_text"],
            "sentiment":    row["sentiment"],
            "confidence":   row["confidence"],
            "cluster_id":   int(row["cluster_id"]),
            "keywords":     row["keywords"],
        }
        for _, row in df.iterrows()
    ]

    output = {
        "status":  "success",
        "total":   len(df),
        "results": results,
        "summary": {
            "positive":                int(sentiment_counts.get("positive", 0)),
            "negative":                int(sentiment_counts.get("negative", 0)),
            "avg_confidence":          round(float(df["confidence"].mean()), 4),
            "n_clusters":              n_clusters_final,
            "silhouette_score":        final_silhouette,
            "confidence_distribution": confidence_distribution,
            "clusters":                cluster_info,
            "topic_overview":          topic_overview,
            "total_input":             total_input,
            "skipped_empty_token":     skipped_count,
        }
    }

    sys.stdout.buffer.write(
        json.dumps(output, ensure_ascii=False).encode("utf-8")
    )


if __name__ == "__main__":
    main()