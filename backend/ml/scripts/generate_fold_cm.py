"""
generate_fold_cm.py — Generate gambar confusion matrix per fold + rata-rata
Membaca data dari fold_analysis.json (output fold_analysis.py)

Cara menjalankan:
  python generate_fold_cm.py

Output: 6 gambar PNG di folder backend/ml/models/
  - CM_Fold1.png sampai CM_Fold5.png
  - CM_RataRata5Fold.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
INPUT_PATH = os.path.join(MODEL_DIR, "fold_analysis.json")


def draw_cm(tn, fp, fn, tp, line1, line2, acc, pre, rec, f1,
            is_avg=False, save_path=None):

    fig, ax = plt.subplots(figsize=(6, 5.8))
    fig.patch.set_facecolor("white")

    cmap   = plt.cm.Blues
    matrix = np.array([[tn, fp], [fn, tp]])
    total  = matrix.sum()
    norm   = mcolors.Normalize(vmin=0, vmax=matrix.max())

    labels_map = [
        ["True Negative (TN)", "False Positive (FP)"],
        ["False Negative (FN)", "True Positive (TP)"],
    ]

    for i in range(2):
        for j in range(2):
            val   = matrix[i, j]
            pct   = val / total * 100
            color = cmap(norm(val))

            ax.add_patch(plt.Rectangle(
                (j, 1 - i), 1, 1,
                facecolor=color, edgecolor="white", linewidth=3
            ))

            brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            txt_color  = "white" if brightness < 0.55 else "#1A237E"

            val_str = f"{val:,.1f}" if is_avg else f"{int(val):,}"
            ax.text(j+0.5, 1-i+0.60, val_str,
                    ha="center", va="center",
                    fontsize=20, fontweight="bold", color=txt_color)
            ax.text(j+0.5, 1-i+0.38, f"({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=11, color=txt_color)
            ax.text(j+0.5, 1-i+0.18, labels_map[i][j],
                    ha="center", va="center",
                    fontsize=9, color=txt_color, fontstyle="italic")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Negatif", "Positif"], fontsize=11, fontweight="bold")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Positif", "Negatif"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Prediksi", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Aktual",   fontsize=12, fontweight="bold", labelpad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"{line1}\n{line2}",
                 fontsize=12, fontweight="bold",
                 color="black", pad=14, linespacing=1.6)

    metrics = (f"Accuracy: {acc}%   |   Precision: {pre}%   |   "
               f"Recall: {rec}%   |   F1-Score: {f1}%")
    fig.text(0.5, 0.01, metrics,
             ha="center", va="bottom", fontsize=8.5, color="#424242",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="#F5F5F5", edgecolor="#BDBDBD", linewidth=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"[SAVED] {os.path.basename(save_path)}")
    plt.close()


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"fold_analysis.json tidak ditemukan: {INPUT_PATH}\n"
            "Jalankan fold_analysis.py terlebih dahulu."
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    folds = data["folds"]
    avg   = data["average"]
    total = data["total_data"]

    print("=" * 55)
    print("  GENERATE CONFUSION MATRIX PER FOLD + RATA-RATA")
    print(f"  Sumber : fold_analysis.json")
    print(f"  Total data : {total:,} ulasan")
    print("=" * 55)

    # generate per fold — n_val diambil dari data masing-masing fold
    for f in folds:
        n_val    = f["n_val"]
        save_path = os.path.join(MODEL_DIR, f"CM_Fold{f['fold']}.png")
        draw_cm(
            tn=f["tn"], fp=f["fp"], fn=f["fn"], tp=f["tp"],
            line1="Confusion Matrix Model Logistic Regression",
            line2=f"(Fold {f['fold']} dari 5, n = {n_val:,} data validasi)",
            acc=f["accuracy"], pre=f["precision"],
            rec=f["recall"],   f1=f["f1_score"],
            is_avg=False,
            save_path=save_path
        )

    # generate rata-rata
    save_path_avg = os.path.join(MODEL_DIR, "CM_RataRata5Fold.png")
    draw_cm(
        tn=avg["tn"], fp=avg["fp"], fn=avg["fn"], tp=avg["tp"],
        line1="Confusion Matrix Model Logistic Regression",
        line2=f"(Rata-Rata 5 Fold, n = {total:,}, std accuracy = \u00b1{avg['std_accuracy']}%)",
        acc=avg["accuracy"], pre=avg["precision"],
        rec=avg["recall"],   f1=avg["f1_score"],
        is_avg=True,
        save_path=save_path_avg
    )

    print("=" * 55)
    print("  Selesai. 6 gambar tersimpan di folder models/")
    print("  - CM_Fold1.png sampai CM_Fold5.png")
    print("  - CM_RataRata5Fold.png")
    print("=" * 55)


if __name__ == "__main__":
    main()