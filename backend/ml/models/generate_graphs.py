"""
generate_graphs.py
Jalankan script ini di folder yang sama dengan model_performance.json
Output: grafik_evaluasi_model.png dan confusion_matrix.png

Cara pakai:
    python generate_graphs.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. LOAD DATA DARI model_performance.json ─────────────────────────────────
# File ini dihasilkan otomatis oleh train.py setelah training selesai
# Lokasinya di backend/models/model_performance.json

PERF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "model_performance.json"
)

if not os.path.exists(PERF_PATH):
    # Fallback: cari di folder yang sama dengan script ini
    PERF_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model_performance.json"
    )

if not os.path.exists(PERF_PATH):
    print(f"[ERROR] File model_performance.json tidak ditemukan.")
    print(f"Pastikan path berikut benar: {PERF_PATH}")
    print("Atau pindahkan script ini ke folder yang sama dengan model_performance.json")
    exit(1)

with open(PERF_PATH, "r", encoding="utf-8") as f:
    perf = json.load(f)

print(f"[INFO] Data berhasil dimuat dari: {PERF_PATH}")

# Ambil nilai metrik
accuracy  = perf["accuracy"]   # sudah dalam persen, misal 86.91
precision = perf["precision"]
recall    = perf["recall"]
f1        = perf["f1_score"]

# Ambil nilai confusion matrix
cm_data = perf["confusion_matrix"]
tn = cm_data["tn"]
fp = cm_data["fp"]
fn = cm_data["fn"]
tp = cm_data["tp"]
total = tn + fp + fn + tp

print(f"[INFO] Accuracy : {accuracy}%")
print(f"[INFO] Precision: {precision}%")
print(f"[INFO] Recall   : {recall}%")
print(f"[INFO] F1-Score : {f1}%")
print(f"[INFO] TN={tn}, FP={fp}, FN={fn}, TP={tp}, Total={total}")

# ── 2. GRAFIK 1: Bar Chart Metrik Evaluasi ───────────────────────────────────

labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#2563EB', '#16A34A', '#D97706', '#DC2626']

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(labels, values, color=colors, width=0.5,
              edgecolor='white', linewidth=0.8)

# Nilai di atas setiap bar
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        f'{val:.2f}%',
        ha='center', va='bottom',
        fontsize=11, fontweight='bold', color='#1e293b'
    )

# Sumbu Y mulai dari 80 agar perbedaan antar metrik terlihat
y_min = 80
y_max = max(values) + 4
ax.set_ylim(y_min, y_max)
ax.set_yticks(np.arange(y_min, y_max, 2))
ax.set_yticklabels([f'{v:.0f}%' for v in np.arange(y_min, y_max, 2)], fontsize=10)

ax.set_xlabel('Metrik Evaluasi', fontsize=12, labelpad=8)
ax.set_ylabel('Nilai (%)', fontsize=12, labelpad=8)
ax.set_title(
    'Hasil Evaluasi Model Logistic Regression',
    fontsize=13, fontweight='bold', pad=14
)

ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out1 = "grafik_evaluasi_model.png"
plt.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Grafik 1 disimpan: {out1}")


# ── 3. GRAFIK 2: Heatmap Confusion Matrix ────────────────────────────────────

cm_array = np.array([
    [tn, fp],   # Aktual Negatif
    [fn, tp],   # Aktual Positif
])

class_labels = ['Negatif', 'Positif']

fig, ax = plt.subplots(figsize=(7, 5.5))

sns.heatmap(
    cm_array,
    annot=False,
    cmap='Blues',
    linewidths=1.5,
    linecolor='white',
    cbar=True,
    ax=ax,
    vmin=0,
    vmax=cm_array.max()
)

# Anotasi manual: nilai, persentase, label
cell_labels = [
    ['True Negative (TN)', 'False Positive (FP)'],
    ['False Negative (FN)', 'True Positive (TP)'],
]

for i in range(2):
    for j in range(2):
        val = cm_array[i, j]
        pct = val / total * 100
        # Deteksi warna teks otomatis:
        # sel terang (nilai kecil) pakai teks gelap, sel gelap pakai teks putih
        text_color = 'white' if val > cm_array.max() * 0.4 else '#1e293b'
        sub_color  = text_color if text_color == 'white' else '#475569'
        ax.text(j + 0.5, i + 0.30, f'{val:,}',
                ha='center', va='center',
                fontsize=15, fontweight='bold', color=text_color)
        ax.text(j + 0.5, i + 0.56, f'({pct:.1f}%)',
                ha='center', va='center',
                fontsize=10, color=text_color, alpha=0.85)
        ax.text(j + 0.5, i + 0.78, cell_labels[i][j],
                ha='center', va='center',
                fontsize=8, color=sub_color, alpha=0.75, style='italic')

ax.set_xlabel('Prediksi', fontsize=12, labelpad=10, fontweight='bold')
ax.set_ylabel('Aktual', fontsize=12, labelpad=10, fontweight='bold')
ax.set_xticklabels(class_labels, fontsize=11)
ax.set_yticklabels(class_labels, fontsize=11, rotation=0)
ax.set_title(
    f'Confusion Matrix Model Logistic Regression\n(n = {total:,} data testing)',
    fontsize=11, fontweight='bold', pad=14
)

plt.tight_layout()
out2 = "confusion_matrix.png"
plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Grafik 2 disimpan: {out2}")

print("\n[SELESAI] Kedua grafik berhasil dibuat.")
print(f"  - {out1}")
print(f"  - {out2}")