# -*- coding: utf-8 -*-
r"""
generate_all_bab4.py
Taruh di   : backend/ml/models/
Jalankan   : python generate_all_bab4.py path/ke/csv_produk.csv

Contoh:
  python generate_all_bab4.py ..\..\storage\app\private\uploads\1779298859_Wardah_Gel_Moisturizer_Series.csv

Output:
  POIN 6 & 7  : poin6_TP.png / TN.png / FP.png / FN.png
                Data: test split 20% dari training data (sama dengan train.py)
  POIN 15 & 16: poin15_aspek_{aspek}_{TP/TN/FP/FN}.png  (20 file)
                poin16_cm_{aspek}.png  (5 file)
                Data: CSV produk yang diinput (Wardah/Skintific/dll)
  POIN 10     : poin10_dendrogram_bab4.png
                poin10_dendrogram_lampiran.png
                Data: CSV produk yang diinput
"""

import os, re, sys, json, joblib, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from collections import Counter
import nltk

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ── PATH ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(_SCRIPT_DIR) if os.path.basename(_SCRIPT_DIR) == "models" else _SCRIPT_DIR
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR     = _SCRIPT_DIR

TRAINING_CSV = os.path.join(DATA_DIR, "training_data.csv")
G2G_CSV      = os.path.join(DATA_DIR, "G2G_dataset.csv")
HEYXI_CSV    = os.path.join(DATA_DIR, "Heyxi_dataset.csv")
KAMUS_PATH   = os.path.join(DATA_DIR, "kamus_alay.json")
BIGRAM_PATH  = os.path.join(MODEL_DIR, "bigram_phraser.pkl")
LR_PATH      = os.path.join(MODEL_DIR, "logistic_regression.pkl")
W2V_PATH     = os.path.join(MODEL_DIR, "vectorizer.pkl")

# ── PREPROCESSING — identik dengan predict.py ────────────────────────────────
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

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    STEMMER_AVAILABLE = True
except ImportError:
    STEMMER_AVAILABLE = False

def load_kamus():
    if os.path.exists(KAMUS_PATH):
        with open(KAMUS_PATH,"r",encoding="utf-8") as f:
            k = json.load(f)
        if "tak" not in k: k["tak"] = "tidak"
        return k
    return {}
KAMUS_ALAY = load_kamus()

# is_noise — identik dengan train.py (versi sederhana, tanpa TOKOPEDIA_ARTIFACTS)
def is_noise(text):
    if not isinstance(text, str): return True
    t = text.strip()
    if len(t) < 5: return True
    ascii_only = t.encode("ascii","ignore").decode("ascii").strip()
    alpha = re.sub(r'[^a-zA-Z]','',ascii_only)
    if len(alpha) < 3:
        return False if len(t) >= 20 else True
    if len(set(alpha.lower())) / len(alpha) < 0.15: return True
    words = ascii_only.split()
    if words and any(len(w) > 25 for w in words): return True
    return False

def preprocess_text(text):
    if not isinstance(text,str) or len(text.strip())<3: return []
    for e,w in EMOJI_MAP.items(): text=text.replace(e,w)
    text = re.sub(r'(.)\1{2,}',r'\1\1',text)
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+","",text)
    text = text.encode("ascii","ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z\s]"," ",text).lower()
    text = " ".join([v for w in text.split() for v in [KAMUS_ALAY.get(w,w)] if v])
    tokens = [t for t in text.split() if t not in STOPWORDS_ID and len(t)>=2]
    result,i = [],0
    while i<len(tokens):
        if tokens[i] in NEGASI_WORDS and i+1<len(tokens):
            result.append(f"tidak_{tokens[i+1]}"); i+=2
        else:
            result.append(tokens[i]); i+=1
    if STEMMER_AVAILABLE:
        result = [t if t.startswith("tidak_") else stemmer.stem(t) for t in result]
    return result

def doc_vector(tokens, model):
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs,axis=0) if vecs else np.zeros(model.vector_size)

TOPIC_WEIGHT = 3.0
TOPIC_RAW_KEYWORDS = {
    "efek_produk":["jerawat","bruntusan","iritasi","gatal","perih","alergi","kulit","wajah","muka",
                   "lembap","lembab","kering","cerah","glowing","cocok","formula","bahan","serum",
                   "bekas","flek","noda","pori","halus","mulus","berminyak","sensitif","breakout",
                   "merah","kemerahan","meradang","moisturizer","pelembap","sunscreen","niacinamide",
                   "ceramide","hyaluronic","tekstur","barrier","kandungan"],
    "pengiriman": ["kirim","sampai","kurir","paket","ekspedisi","pengiriman","lambat","lama",
                   "ongkir","ongkos","resi","jnt","jne","sicepat","anteraja","gosend",
                   "tiba","datang","estimasi","tracking","cepat"],
    "harga_nilai":["harga","mahal","murah","worth","sepadan","promo","diskon","hemat",
                   "terjangkau","budget","ekonomis","cashback","voucher","sale","harganya"],
    "kemasan":    ["kemasan","packing","packaging","botol","tube","pot","pump","wadah",
                   "bocor","pecah","rusak","segel","kardus","bubble","wrap","rapi","rapih","kemas"],
    "pembelian":  ["palsu","kw","ori","original","asli","seller","toko","reseller","official",
                   "repeat","ulang","restock","langganan","respon","beli"],
}
TOPIC_LABELS = {
    "efek_produk":"Efek & Kecocokan Produk",
    "pengiriman": "Layanan Pengiriman",
    "harga_nilai":"Harga Produk",
    "kemasan":    "Kondisi Kemasan",
    "pembelian":  "Layanan Toko & Keaslian",
}
ALL_TOPIC_WORDS = set(w for ws in TOPIC_RAW_KEYWORDS.values() for w in ws)
ASPECT_COLORS = {
    "efek_produk":"#2563EB","pengiriman":"#16A34A",
    "harga_nilai":"#D97706","kemasan":"#7C3AED","pembelian":"#0891B2",
}

def detect_topic(text):
    if not isinstance(text,str): return []
    t = text.lower()
    return [k for k,ws in TOPIC_RAW_KEYWORDS.items() if any(w in t for w in ws)]

def topic_weighted_vector(tokens, model):
    vectors,weights = [],[]
    for t in tokens:
        if t in model.wv:
            w = TOPIC_WEIGHT if t.replace("tidak_","") in ALL_TOPIC_WORDS else 1.0
            vectors.append(model.wv[t]); weights.append(w)
    if not vectors: return np.zeros(model.vector_size)
    return np.average(np.array(vectors),axis=0,weights=np.array(weights))

def load_csv_file(path):
    for sep in [";",","]:
        try:
            tmp = pd.read_csv(path,sep=sep,encoding="utf-8-sig")
            if len(tmp.columns)>1: return tmp
        except: continue
    return pd.read_csv(path,encoding="utf-8-sig")

def find_col(df, candidates):
    cm = {c.lower():c for c in df.columns}
    for c in candidates:
        if c in cm: return cm[c]
    return None

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"      OK -> {name}")

# ── KONSTANTA WARNA KATEGORI ──────────────────────────────────────────────────
CAT_COLOR  = {"TP":"#16A34A","TN":"#2563EB","FP":"#D97706","FN":"#DC2626"}
CAT_BG     = {"TP":"#F0FDF4","TN":"#EFF6FF","FP":"#FFFBEB","FN":"#FFF5F5"}
CAT_BORDER = {"TP":"#86EFAC","TN":"#93C5FD","FP":"#FCD34D","FN":"#FCA5A5"}
CAT_AKTUAL = {"TP":"Positif","TN":"Negatif","FP":"Negatif","FN":"Positif"}
CAT_PRED   = {"TP":"Positif","TN":"Negatif","FP":"Positif","FN":"Negatif"}
CAT_TITLE  = {
    "TP":"True Positive (TP)  v",
    "TN":"True Negative (TN)  v",
    "FP":"False Positive (FP)  x",
    "FN":"False Negative (FN)  x",
}

def make_sample_page(cat_key, subtitle, samples, n_total):
    """1 halaman penuh per kategori — font besar, terbaca."""
    n     = len(samples)
    fig_h = 3.5 + n * 1.7
    fig,ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor('white')
    color  = CAT_COLOR[cat_key]
    bg     = CAT_BG[cat_key]
    border = CAT_BORDER[cat_key]
    ax.set_facecolor(bg); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

    header_h  = 0.13
    col_hdr_h = 0.055
    content_h = 1.0 - header_h - col_hdr_h - 0.04
    row_h     = content_h / max(n, 1)

    # Header
    ax.add_patch(mpatches.FancyBboxPatch((0,1-header_h),1,header_h,
        boxstyle="square,pad=0",facecolor=color,edgecolor='none',
        transform=ax.transAxes,zorder=3))
    ax.text(0.5,1-header_h*0.28,CAT_TITLE[cat_key],
            transform=ax.transAxes,ha='center',va='center',
            fontsize=20,fontweight='bold',color='white',zorder=4)
    ax.text(0.5,1-header_h*0.68,subtitle,
            transform=ax.transAxes,ha='center',va='center',
            fontsize=12,color='white',alpha=0.93,zorder=4)
    ax.text(0.5,1-header_h*0.95,
            f"Aktual: {CAT_AKTUAL[cat_key]}   ->   Prediksi: {CAT_PRED[cat_key]}"
            f"   |   Total kategori ini: {n_total:,} ulasan   |   Menampilkan {n} sampel",
            transform=ax.transAxes,ha='center',va='top',
            fontsize=10.5,color=color,fontstyle='italic')

    # Column headers
    y_col = 1 - header_h - col_hdr_h*0.45 - 0.025
    for cx,cw,lab in [(0.005,0.05,'No'),
                      (0.06,0.79,'Teks Ulasan (asli dari data)'),
                      (0.855,0.14,'Label\nAktual')]:
        ax.add_patch(mpatches.FancyBboxPatch((cx,y_col-col_hdr_h*0.5),cw,col_hdr_h,
            boxstyle="round,pad=0.003",facecolor=color,alpha=0.18,
            edgecolor='none',transform=ax.transAxes))
        ax.text(cx+cw/2,y_col,lab,transform=ax.transAxes,
                ha='center',va='center',fontsize=11.5,fontweight='bold',color=color)

    # Rows
    y_start = 1 - header_h - col_hdr_h - 0.045
    for ri,text in enumerate(samples):
        y = y_start - ri*row_h
        if ri%2==0:
            ax.add_patch(mpatches.FancyBboxPatch((0.003,y-row_h*0.83),0.994,row_h*0.90,
                boxstyle="round,pad=0.003",facecolor=color,alpha=0.07,
                edgecolor=border,linewidth=0.6,transform=ax.transAxes))
        ax.text(0.028,y-row_h*0.38,f'{ri+1}',
                transform=ax.transAxes,ha='center',va='center',
                fontsize=13,fontweight='bold',color=color)
        words = text.split(); lines,cur = [],''
        for w in words:
            if len(cur)+len(w)+1 <= 90: cur += (' ' if cur else '')+w
            else: lines.append(cur); cur=w
        if cur: lines.append(cur)
        lines = lines[:3]
        y_text = y - row_h*0.16
        for li,line in enumerate(lines):
            ax.text(0.065,y_text-li*row_h*0.29,line,
                    transform=ax.transAxes,ha='left',va='top',
                    fontsize=11.5,color='#1f2937')
        lc = '#16A34A' if CAT_AKTUAL[cat_key]=='Positif' else '#DC2626'
        ax.text(0.927,y-row_h*0.38,CAT_AKTUAL[cat_key],
                transform=ax.transAxes,ha='center',va='center',
                fontsize=11.5,fontweight='bold',color=lc)

    ax.add_patch(mpatches.FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0.008",
        facecolor='none',edgecolor=border,linewidth=2.5,transform=ax.transAxes))
    return fig

def get_samples(df_subset, n=10):
    s = df_subset.copy()
    s["tl"] = s["review_text"].str.len()
    pool = s[(s["tl"]>=25)&(s["tl"]<=180)]
    if len(pool)<n: pool = s[s["tl"]>=15]
    if len(pool)<n: pool = s
    return pool.sample(min(n,len(pool)),random_state=42)["review_text"].tolist()

# =============================================================================
print("="*62)
print("  generate_all_bab4.py")
print("="*62)

if len(sys.argv) < 2:
    print("\n[ERROR] Sertakan path CSV produk.")
    print("Contoh:")
    print("  python generate_all_bab4.py ..\\..\\storage\\app\\private\\uploads\\NamaFile.csv")
    sys.exit(1)

CSV_PRODUK = sys.argv[1]
if not os.path.exists(CSV_PRODUK):
    print(f"\n[ERROR] File tidak ditemukan: {CSV_PRODUK}")
    sys.exit(1)

# ── Load model ────────────────────────────────────────────────────────────────
print("\n[LOAD] Model...")
w2v_model = joblib.load(W2V_PATH)
lr_model  = joblib.load(LR_PATH)
phraser   = joblib.load(BIGRAM_PATH) if os.path.exists(BIGRAM_PATH) else None
print("       OK")

# =============================================================================
# POIN 6 & 7 — Test split 20% dari training data (identik dengan train.py)
# =============================================================================
print("\n[POIN 6&7] Load training data untuk test split...")
frames = []
if os.path.exists(TRAINING_CSV):
    df_main = pd.read_csv(TRAINING_CSV)
    if "source" not in df_main.columns: df_main["source"] = "sekarmulyani"
    frames.append(df_main)
for path,src in [(G2G_CSV,"g2g"),(HEYXI_CSV,"heyxi")]:
    if not os.path.exists(path): continue
    for sep in [",",";"]:
        try:
            tmp = pd.read_csv(path,sep=sep,encoding="utf-8-sig")
            if len(tmp.columns)<=1: continue
            rows=[]
            for _,row in tmp.iterrows():
                text=str(row.get("Comment","")).strip()
                try: rating=float(row.get("Rating",0))
                except: continue
                if rating<=2: label="negative"
                elif rating>=4: label="positive"
                else: continue
                if len(text)>=5: rows.append({"review_text":text,"label":label,"source":src})
            if rows: frames.append(pd.DataFrame(rows))
            break
        except: continue

df_train = pd.concat(frames,ignore_index=True)
df_train = df_train.dropna(subset=["review_text","label"])
df_train = df_train[df_train["label"].isin(["positive","negative"])].drop_duplicates(subset=["review_text"])
before   = len(df_train)
df_train = df_train[~df_train["review_text"].apply(is_noise)].reset_index(drop=True)
print(f"       Total bersih: {len(df_train):,} (noise dibuang: {before-len(df_train):,})")

print("       Preprocessing training data...")
df_train["tokens"] = df_train["review_text"].apply(preprocess_text)
if phraser:
    df_train["tokens"] = df_train["tokens"].apply(lambda t: list(phraser[t]))
df_train = df_train[df_train["tokens"].map(len)>0].reset_index(drop=True)

X_all = np.array([doc_vector(t,w2v_model) for t in df_train["tokens"]])
y_all = df_train["label"].values

# Split identik dengan train.py: seed=42, stratify
X_tr,X_te,y_tr,y_te,idx_tr,idx_te = train_test_split(
    X_all, y_all, df_train.index.tolist(),
    test_size=0.2, random_state=42, stratify=y_all
)
df_test = df_train.loc[idx_te].copy().reset_index(drop=True)
df_test["pred"]            = lr_model.predict(X_te)
df_test["detected_topics"] = df_test["review_text"].apply(detect_topic)

TP_m = (df_test["label"]=="positive")&(df_test["pred"]=="positive")
TN_m = (df_test["label"]=="negative")&(df_test["pred"]=="negative")
FP_m = (df_test["label"]=="negative")&(df_test["pred"]=="positive")
FN_m = (df_test["label"]=="positive")&(df_test["pred"]=="negative")
TP,TN,FP,FN = TP_m.sum(),TN_m.sum(),FP_m.sum(),FN_m.sum()
print(f"       Test set: {len(df_test):,} | TP={TP:,} TN={TN:,} FP={FP:,} FN={FN:,}")

for cat_key,mask,count in [("TP",TP_m,TP),("TN",TN_m,TN),("FP",FP_m,FP),("FN",FN_m,FN)]:
    if count==0:
        print(f"       [SKIP] poin6_{cat_key}.png")
        continue
    samples  = get_samples(df_test[mask], n=10)
    subtitle = (f"Data: Test Set 20% (seed=42) dari Training Data"
                f"   |   Total test set: {len(df_test):,} ulasan")
    fig = make_sample_page(cat_key, subtitle, samples, count)
    save(fig, f"poin6_{cat_key}.png")

# =============================================================================
# POIN 15 & 16 — CSV produk (Wardah/Skintific/dll)
# =============================================================================
print(f"\n[POIN 15&16] Load CSV produk: {os.path.basename(CSV_PRODUK)}")
df_prod = load_csv_file(CSV_PRODUK)
text_col   = find_col(df_prod,["comment","review_text","review","ulasan","text","komentar"])
rating_col = find_col(df_prod,["rating","bintang","star","score"])
prod_col   = find_col(df_prod,["productname","product_name","product","produk","nama_produk"])

if not text_col: text_col = df_prod.columns[0]
df_prod["review_text"]  = df_prod[text_col].astype(str)
df_prod["product_name"] = df_prod[prod_col].astype(str) if prod_col else "Produk Skincare"
df_prod = df_prod.dropna(subset=["review_text"]).reset_index(drop=True)

# Noise filter identik dengan predict.py (versi train.py yang sederhana)
df_prod = df_prod[~df_prod["review_text"].apply(is_noise)].reset_index(drop=True)

prod_name = df_prod["product_name"].mode()[0]
print(f"       Produk : {prod_name}")
print(f"       Ulasan : {len(df_prod)}")

# Label dari rating
has_label = False
if rating_col:
    def to_label(r):
        try:
            rv = float(r)
            if rv<=2: return "negative"
            if rv>=4: return "positive"
        except: pass
        return None
    df_prod["label"] = df_prod[rating_col].apply(to_label)
    df_lbl = df_prod[df_prod["label"].notna()].copy().reset_index(drop=True)
    if len(df_lbl) > 0:
        has_label = True
        print(f"       Berlabel: {len(df_lbl)} "
              f"(pos={( df_lbl['label']=='positive').sum()} "
              f"neg={(df_lbl['label']=='negative').sum()})")

if not has_label:
    print("       [WARN] Tidak ada kolom rating — poin 15 & 16 dilewati.")
else:
    # Preprocessing & predict
    df_lbl["tokens"] = df_lbl["review_text"].apply(preprocess_text)
    if phraser:
        df_lbl["tokens"] = df_lbl["tokens"].apply(lambda t: list(phraser[t]))
    df_lbl = df_lbl[df_lbl["tokens"].map(len)>0].reset_index(drop=True)
    X_lbl  = np.array([doc_vector(t,w2v_model) for t in df_lbl["tokens"]])
    df_lbl["pred"]            = lr_model.predict(X_lbl)
    df_lbl["detected_topics"] = df_lbl["review_text"].apply(detect_topic)

    pTP = (df_lbl["label"]=="positive")&(df_lbl["pred"]=="positive")
    pTN = (df_lbl["label"]=="negative")&(df_lbl["pred"]=="negative")
    pFP = (df_lbl["label"]=="negative")&(df_lbl["pred"]=="positive")
    pFN = (df_lbl["label"]=="positive")&(df_lbl["pred"]=="negative")
    print(f"       TP={pTP.sum()} TN={pTN.sum()} FP={pFP.sum()} FN={pFN.sum()}")

    # ── Poin 15: 20 file (5 aspek x 4 kategori) ──────────────────────────────
    masks_prod = {"TP":pTP,"TN":pTN,"FP":pFP,"FN":pFN}
    for topic_key,topic_label in TOPIC_LABELS.items():
        mask_topic = df_lbl["detected_topics"].apply(lambda t: topic_key in t)
        sub_topic  = df_lbl[mask_topic].reset_index(drop=True)
        n_topic    = len(sub_topic)
        if n_topic == 0:
            print(f"       [SKIP] poin15_{topic_key}_*.png — aspek tidak ada data")
            continue

        for cat_key in ["TP","TN","FP","FN"]:
            if cat_key=="TP":   cmask=(sub_topic["label"]=="positive")&(sub_topic["pred"]=="positive")
            elif cat_key=="TN": cmask=(sub_topic["label"]=="negative")&(sub_topic["pred"]=="negative")
            elif cat_key=="FP": cmask=(sub_topic["label"]=="negative")&(sub_topic["pred"]=="positive")
            else:               cmask=(sub_topic["label"]=="positive")&(sub_topic["pred"]=="negative")

            n_cat = cmask.sum()
            fname = f"poin15_{topic_key}_{cat_key}.png"
            if n_cat == 0:
                print(f"       [SKIP] {fname} — tidak ada data")
                continue

            pool = sub_topic[cmask].copy()
            pool["tl"] = pool["review_text"].str.len()
            use = pool[(pool["tl"]>=25)&(pool["tl"]<=180)]
            if len(use)<5: use=pool
            samples = use.sample(min(5,len(use)),random_state=42)["review_text"].tolist()

            subtitle = (f"Aspek: {topic_label}   |   Produk: {prod_name}"
                        f"   |   Ulasan aspek ini: {n_topic:,}   |   Kategori ini: {n_cat:,}")
            fig = make_sample_page(cat_key, subtitle, samples, n_cat)
            save(fig, fname)

    # ── Poin 16: 5 file confusion matrix per aspek ───────────────────────────
    for topic_key,topic_label in TOPIC_LABELS.items():
        mask_topic = df_lbl["detected_topics"].apply(lambda t: topic_key in t)
        sub   = df_lbl[mask_topic]
        n_asp = len(sub)
        if n_asp == 0:
            print(f"       [SKIP] poin16_cm_{topic_key}.png — tidak ada data")
            continue

        aTP = int(((sub["label"]=="positive")&(sub["pred"]=="positive")).sum())
        aTN = int(((sub["label"]=="negative")&(sub["pred"]=="negative")).sum())
        aFP = int(((sub["label"]=="negative")&(sub["pred"]=="positive")).sum())
        aFN = int(((sub["label"]=="positive")&(sub["pred"]=="negative")).sum())
        total = aTP+aTN+aFP+aFN
        if total == 0: continue

        acc  = (aTP+aTN)/total*100
        prec = aTP/(aTP+aFP)*100 if (aTP+aFP)>0 else 0
        rec  = aTP/(aTP+aFN)*100 if (aTP+aFN)>0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        asp_color = ASPECT_COLORS[topic_key]

        fig,axes = plt.subplots(1,2,figsize=(16,8),gridspec_kw={'width_ratios':[1.2,1]})
        fig.patch.set_facecolor('white')
        fig.suptitle(
            f"Confusion Matrix & Evaluasi  -  Aspek: {topic_label}\n"
            f"(n={n_asp:,} ulasan dari {prod_name})",
            fontsize=14,fontweight='bold',y=0.98,color='#1e293b')

        ax_cm = axes[0]
        cm = np.array([[aTN,aFP],[aFN,aTP]])
        sns.heatmap(cm,annot=False,cmap='Blues',linewidths=2,linecolor='white',
                    cbar=True,ax=ax_cm,vmin=0,vmax=max(cm.max(),1))
        for i in range(2):
            for j in range(2):
                v   = [[aTN,aFP],[aFN,aTP]][i][j]
                lbl = [['True Negative','False Positive'],['False Negative','True Positive']][i][j]
                pct = v/total*100
                tc  = 'white' if cm.max()>0 and v>cm.max()*0.4 else '#1e293b'
                sc  = 'white' if tc=='white' else '#6b7280'
                ax_cm.text(j+0.5,i+0.25,f'{v:,}',ha='center',va='center',
                           fontsize=18,fontweight='bold',color=tc)
                ax_cm.text(j+0.5,i+0.52,f'({pct:.1f}%)',ha='center',va='center',
                           fontsize=12,color=tc,alpha=0.9)
                ax_cm.text(j+0.5,i+0.76,lbl,ha='center',va='center',
                           fontsize=10,color=sc,style='italic')
        ax_cm.set_xticklabels(['Prediksi\nNegatif','Prediksi\nPositif'],fontsize=12)
        ax_cm.set_yticklabels(['Aktual\nNegatif','Aktual\nPositif'],fontsize=12,rotation=0)
        ax_cm.set_xlabel('Prediksi',fontsize=12,labelpad=8)
        ax_cm.set_ylabel('Aktual',fontsize=12,labelpad=8)
        ax_cm.set_title('Confusion Matrix',fontsize=13,fontweight='bold',pad=10)

        ax_bar = axes[1]
        bars = ax_bar.barh(['Accuracy','Precision','Recall','F1-Score'],
                           [acc,prec,rec,f1],
                           color=['#2563EB','#16A34A','#D97706','#DC2626'],
                           height=0.5,edgecolor='white')
        for bar,val in zip(bars,[acc,prec,rec,f1]):
            ax_bar.text(min(val+0.8,103),bar.get_y()+bar.get_height()/2,
                        f'{val:.2f}%',va='center',ha='left',
                        fontsize=12,fontweight='bold',color='#1e293b')
        ax_bar.set_xlim(0,112)
        ax_bar.set_xlabel('Nilai (%)',fontsize=12)
        ax_bar.set_title('Metrik Evaluasi',fontsize=13,fontweight='bold',pad=10)
        ax_bar.set_facecolor('#FAFAFA')
        ax_bar.grid(True,linestyle='--',alpha=0.4,axis='x',color='#CBD5E1')
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.text(0.5,0.12,
            f"Accuracy  : {acc:.2f}%\nPrecision : {prec:.2f}%\n"
            f"Recall    : {rec:.2f}%\nF1-Score  : {f1:.2f}%\n\n"
            f"TP={aTP:,}  TN={aTN:,}\nFP={aFP:,}  FN={aFN:,}",
            transform=ax_bar.transAxes,ha='center',va='bottom',fontsize=11,
            fontfamily='monospace',color='#1e293b',
            bbox=dict(boxstyle='round,pad=0.5',facecolor='#F8FAFC',
                      edgecolor=asp_color,lw=1.5))
        plt.tight_layout(rect=[0,0,1,0.95])
        save(fig, f"poin16_cm_{topic_key}.png")

# =============================================================================
# POIN 10 — Dendrogram dari CSV produk
# =============================================================================
print(f"\n[POIN 10] Dendrogram dari {os.path.basename(CSV_PRODUK)}...")

# Preprocessing ulang dari df_prod (semua ulasan, tidak perlu label)
df_prod["tokens"] = df_prod["review_text"].apply(preprocess_text)
if phraser:
    df_prod["tokens"] = df_prod["tokens"].apply(lambda t: list(phraser[t]))
df_prod = df_prod[df_prod["tokens"].map(len)>=3].reset_index(drop=True)

X_sent  = np.array([doc_vector(t,w2v_model) for t in df_prod["tokens"]])
X_clust = np.array([topic_weighted_vector(t,w2v_model) for t in df_prod["tokens"]])
X_clust = normalize(X_clust,norm='l2')
df_prod["pred_sent"] = lr_model.predict(X_sent)
df_prod["topic"]     = df_prod["review_text"].apply(
    lambda x: detect_topic(x)[0] if detect_topic(x) else "umum"
)

n_d = len(df_prod)
max_k = min(8,max(2,int(np.sqrt(n_d/2))))
best_k,best_sil = 2,-1.0
for k in range(2,min(max_k+1,n_d//10+2)):
    try:
        lbl=AgglomerativeClustering(n_clusters=k,linkage="average").fit_predict(X_clust)
        if len(set(lbl))<2: continue
        s=silhouette_score(X_clust,lbl,sample_size=min(500,n_d),random_state=42)
        if s>best_sil: best_sil,best_k=s,k
    except: continue

labels_ahc = AgglomerativeClustering(n_clusters=best_k,linkage="average").fit_predict(X_clust)
df_prod["cluster_id"] = labels_ahc
print(f"       k={best_k}  silhouette={best_sil:.4f}")

# Subset proporsional max 60
MAX_D,sub_idx = 60,[]
for cid in range(best_k):
    m = df_prod["cluster_id"]==cid
    n_pick = max(3,int(MAX_D*m.sum()/n_d))
    sub_idx.extend(df_prod[m].sample(min(n_pick,m.sum()),random_state=42).index.tolist())
sub_idx = list(dict.fromkeys(sub_idx))[:MAX_D]
df_sub  = df_prod.loc[sub_idx].reset_index(drop=True)
X_sub   = X_clust[sub_idx]

Z         = linkage(X_sub,method='average',metric='euclidean')
threshold = Z[:,2].max()*0.50
CCOLORS   = [matplotlib.colors.to_hex(c) for c in plt.cm.tab10(np.linspace(0,0.9,best_k))]
TSHORT    = {"efek_produk":"Efek","pengiriman":"Pengiriman",
             "harga_nilai":"Harga","kemasan":"Kemasan","pembelian":"Toko","umum":"Umum"}

leaf_labels = [
    f"C{int(df_sub.loc[i,'cluster_id'])+1} | "
    f"{TSHORT.get(df_sub.loc[i,'topic'],'Umum')} "
    f"[{'+'if df_sub.loc[i,'pred_sent']=='positive' else '-'}]"
    for i in range(len(df_sub))
]
legend_h = [mpatches.Patch(color=CCOLORS[i],label=f'Cluster {i+1}') for i in range(best_k)]
legend_h += [plt.Line2D([0],[0],color='#DC2626',linestyle='--',lw=2,
                         label=f'Threshold ({threshold:.3f})')]

def plot_dendro(ax, title):
    dendrogram(Z,ax=ax,labels=leaf_labels,orientation='right',
               color_threshold=threshold,above_threshold_color='#94A3B8',
               leaf_font_size=9,distance_sort='ascending')
    ax.axvline(x=threshold,color='#DC2626',linestyle='--',linewidth=2)
    ax.set_title(title,fontsize=11,fontweight='bold',pad=10)
    ax.set_xlabel('Euclidean Distance (L2-normalized)',fontsize=10)
    ax.set_ylabel('Ulasan (Cluster | Aspek | Sentimen)',fontsize=10)
    ax.set_facecolor('#FAFAFA')
    ax.grid(True,linestyle='--',alpha=0.3,axis='x',color='#CBD5E1')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(handles=legend_h,loc='lower right',fontsize=9,
              framealpha=0.95,edgecolor='#CBD5E1')
    ax.text(0.98,0.01,
        f"Cluster optimal : {best_k}\nSilhouette Score : {best_sil:.4f}\n"
        f"Metode           : Average Linkage\nJarak            : Euclidean (L2)",
        transform=ax.transAxes,ha='right',va='bottom',fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5',facecolor='#EFF6FF',edgecolor='#93C5FD',lw=1))

fig_h = max(12,len(df_sub)*0.25)
fig,ax = plt.subplots(figsize=(15,fig_h))
fig.patch.set_facecolor('white')
plot_dendro(ax,
    f"Dendrogram Agglomerative Hierarchical Clustering\n"
    f"Average Linkage - Euclidean Distance  |  Produk: {prod_name}  |  "
    f"Subset {len(df_sub)} dari {len(df_prod)} ulasan")
plt.tight_layout()
save(fig,"poin10_dendrogram_bab4.png")

fig_h2 = max(16,len(df_sub)*0.30)
fig,ax2 = plt.subplots(figsize=(18,fig_h2))
fig.patch.set_facecolor('white')
plot_dendro(ax2,
    f"Lampiran - Dendrogram Agglomerative Hierarchical Clustering\n"
    f"Produk: {prod_name}  |  Subset {len(df_sub)} ulasan  |  "
    f"k={best_k}  |  Silhouette={best_sil:.4f}")
plt.tight_layout()
save(fig,"poin10_dendrogram_lampiran.png")

print(f"""
{'='*62}
  SELESAI!  File tersimpan di: {OUT_DIR}

  Poin 6 & 7  : poin6_TP / TN / FP / FN .png
                (test split 20% training data)
  Poin 15     : poin15_{{aspek}}_{{TP/TN/FP/FN}}.png
                (data produk: {os.path.basename(CSV_PRODUK)})
  Poin 16     : poin16_cm_{{aspek}}.png
                (data produk: {os.path.basename(CSV_PRODUK)})
  Poin 10     : poin10_dendrogram_bab4.png
                poin10_dendrogram_lampiran.png
                (data produk: {os.path.basename(CSV_PRODUK)})
{'='*62}
""")