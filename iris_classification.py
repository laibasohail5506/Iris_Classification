
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── Palette ──────────────────────────────────────────────────
DARK_BG   = "#042C53"
MID_BG    = "#0C447C"
CARD_BG   = "#185FA5"
ACCENT    = "#EF9F27"
ACCENT2   = "#5DCAA5"
TEXT_MAIN = "#E6F1FB"
TEXT_DIM  = "#85B7EB"
C_SET     = "#5DCAA5"
C_VER     = "#EF9F27"
C_VIR     = "#E24B4A"

# ── Train model ──────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_tr, y_tr)
preds = model.predict(X_te)
acc   = accuracy_score(y_te, preds)
f1    = f1_score(y_te, preds, average="weighted")
cm    = confusion_matrix(y_te, preds)

# ── Canvas ───────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=DARK_BG)
gs  = GridSpec(
    3, 3,
    figure=fig,
    left=0.05, right=0.97,
    top=0.88,  bottom=0.07,
    hspace=0.55, wspace=0.38,
)

# ════════════════════════════════════════════════════════════
#  HEADER BAND
# ════════════════════════════════════════════════════════════
fig.add_axes([0, 0.89, 1, 0.11], facecolor=MID_BG)

fig.text(0.50, 0.945, "Data Classification Using AI",
         ha="center", va="center",
         fontsize=22, fontweight="bold", color=TEXT_MAIN)

fig.text(0.50, 0.905,
         "K-Nearest Neighbors · Iris Dataset · Supervised Learning  |  DecodeLabs · Batch 2026",
         ha="center", va="center",
         fontsize=10, color=TEXT_DIM)

# Left gold bar
bar_ax = fig.add_axes([0, 0.89, 0.006, 0.11], facecolor=ACCENT)

# ════════════════════════════════════════════════════════════
#  PANEL 1 — Scatter: Petal features  (row 0, col 0)
# ════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(MID_BG)
for spine in ax1.spines.values():
    spine.set_edgecolor(CARD_BG)

colors_map = [C_SET, C_VER, C_VIR]
markers_map = ["o", "s", "^"]
labels_map  = ["Setosa", "Versicolor", "Virginica"]

for cls, col, mk, lbl in zip([0,1,2], colors_map, markers_map, labels_map):
    mask = y == cls
    ax1.scatter(iris.data[mask, 2], iris.data[mask, 3],
                c=col, marker=mk, s=28, alpha=0.85,
                label=lbl, linewidths=0)

ax1.set_xlabel("Petal length (cm)", fontsize=8, color=TEXT_DIM)
ax1.set_ylabel("Petal width (cm)",  fontsize=8, color=TEXT_DIM)
ax1.set_title("Iris feature space", fontsize=9,
              fontweight="bold", color=TEXT_MAIN, pad=6)
ax1.tick_params(colors=TEXT_DIM, labelsize=7)
ax1.legend(fontsize=7, facecolor=DARK_BG, edgecolor=CARD_BG,
           labelcolor=TEXT_MAIN, markerscale=0.9,
           loc="upper left", framealpha=0.9)

# ════════════════════════════════════════════════════════════
#  PANEL 2 — Elbow curve  (row 0, col 1)
# ════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(MID_BG)
for spine in ax2.spines.values():
    spine.set_edgecolor(CARD_BG)

k_range    = range(1, 21)
error_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr, y_tr)
    error_list.append(1 - accuracy_score(y_te, knn.predict(X_te)))

opt_k = int(np.argmin(error_list)) + 1

ax2.plot(k_range, error_list, color=TEXT_DIM,
         linewidth=1.6, marker="o", markersize=4,
         markerfacecolor=TEXT_DIM)
ax2.axvline(opt_k, color=ACCENT, linewidth=1.4,
            linestyle="--", label=f"Optimal K = {opt_k}")
ax2.scatter([opt_k], [error_list[opt_k-1]],
            color=ACCENT, s=70, zorder=5)

ax2.set_xlabel("K value",     fontsize=8, color=TEXT_DIM)
ax2.set_ylabel("Error rate",  fontsize=8, color=TEXT_DIM)
ax2.set_title("Elbow — optimal K", fontsize=9,
              fontweight="bold", color=TEXT_MAIN, pad=6)
ax2.tick_params(colors=TEXT_DIM, labelsize=7)
ax2.legend(fontsize=7, facecolor=DARK_BG, edgecolor=CARD_BG,
           labelcolor=TEXT_MAIN, framealpha=0.9)

# ════════════════════════════════════════════════════════════
#  PANEL 3 — Confusion matrix  (row 0, col 2)
# ════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(MID_BG)

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax3.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

for i in range(3):
    for j in range(3):
        val = cm[i, j]
        col = "white" if cm_norm[i, j] > 0.55 else TEXT_MAIN
        ax3.text(j, i, str(val), ha="center", va="center",
                 fontsize=11, fontweight="bold", color=col)

tick_lbl = ["Setosa", "Versicolor", "Virginica"]
ax3.set_xticks([0, 1, 2])
ax3.set_yticks([0, 1, 2])
ax3.set_xticklabels(tick_lbl, fontsize=7, color=TEXT_DIM, rotation=15)
ax3.set_yticklabels(tick_lbl, fontsize=7, color=TEXT_DIM)
ax3.set_xlabel("Predicted", fontsize=8, color=TEXT_DIM)
ax3.set_ylabel("Actual",    fontsize=8, color=TEXT_DIM)
ax3.set_title("Confusion matrix", fontsize=9,
              fontweight="bold", color=TEXT_MAIN, pad=6)
for spine in ax3.spines.values():
    spine.set_edgecolor(CARD_BG)

# ════════════════════════════════════════════════════════════
#  PANEL 4 — F1 per class bar  (row 1, col 0)
# ════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(MID_BG)
for spine in ax4.spines.values():
    spine.set_edgecolor(CARD_BG)

f1_cls = f1_score(y_te, preds, average=None)
bars = ax4.bar(["Setosa", "Versicolor", "Virginica"],
               f1_cls,
               color=[C_SET, C_VER, C_VIR],
               width=0.5, zorder=3)
ax4.set_ylim(0, 1.18)
ax4.set_title("F1 score per class", fontsize=9,
              fontweight="bold", color=TEXT_MAIN, pad=6)
ax4.set_ylabel("F1 score", fontsize=8, color=TEXT_DIM)
ax4.tick_params(colors=TEXT_DIM, labelsize=7)
ax4.yaxis.grid(True, color=CARD_BG, linewidth=0.5, zorder=0)
ax4.set_axisbelow(True)

for bar, val in zip(bars, f1_cls):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.03,
             f"{val:.2f}", ha="center", va="bottom",
             fontsize=9, fontweight="bold", color=TEXT_MAIN)

# ════════════════════════════════════════════════════════════
#  PANEL 5 — KNN decision boundary  (row 1, col 1)
# ════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(MID_BG)
for spine in ax5.spines.values():
    spine.set_edgecolor(CARD_BG)

feat0, feat1 = 2, 3
X2_all = X_sc[:, [feat0, feat1]]
X2_tr, X2_te, y2_tr, y2_te = train_test_split(
    X2_all, y, test_size=0.2, random_state=42, stratify=y)
knn2d = KNeighborsClassifier(n_neighbors=5)
knn2d.fit(X2_tr, y2_tr)

x_min, x_max = X2_all[:,0].min()-0.5, X2_all[:,0].max()+0.5
y_min, y_max = X2_all[:,1].min()-0.5, X2_all[:,1].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220),
                     np.linspace(y_min, y_max, 220))
Z = knn2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

region_colors = np.array([[4,44,83],[30,94,165],[24,94,117]])
rgb_img = region_colors[Z] / 255.0
ax5.imshow(rgb_img, extent=[x_min,x_max,y_min,y_max],
           origin="lower", aspect="auto", alpha=0.55)

for cls, col, mk in zip([0,1,2], colors_map, markers_map):
    mask = y2_tr == cls
    ax5.scatter(X2_tr[mask,0], X2_tr[mask,1],
                c=col, marker=mk, s=22, alpha=0.9, linewidths=0)

ax5.set_xlabel("Petal length (scaled)", fontsize=8, color=TEXT_DIM)
ax5.set_ylabel("Petal width (scaled)",  fontsize=8, color=TEXT_DIM)
ax5.set_title("Decision boundary (K=5)", fontsize=9,
              fontweight="bold", color=TEXT_MAIN, pad=6)
ax5.tick_params(colors=TEXT_DIM, labelsize=7)

# ════════════════════════════════════════════════════════════
#  PANEL 6 — Pipeline diagram  (row 1–2, col 2)
# ════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[1:, 2])
ax6.set_facecolor(DARK_BG)
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.axis("off")

steps = [
    ("01  Load Data",    "150 samples · 4 features",  0.83, CARD_BG,  TEXT_MAIN),
    ("02  Scale",        "StandardScaler  μ=0 σ=1",   0.63, CARD_BG,  TEXT_MAIN),
    ("03  Split",        "80% train · 20% test",       0.43, CARD_BG,  TEXT_MAIN),
    ("04  KNN (K=5)",    "Proximity principle",        0.23, ACCENT,   "#412402"),
    ("05  Evaluate",     "F1 + confusion matrix",      0.05, CARD_BG,  TEXT_MAIN),
]

for title, sub, cy, bg, fg in steps:
    box = FancyBboxPatch((0.08, cy-0.08), 0.84, 0.155,
                         boxstyle="round,pad=0.015",
                         facecolor=bg, edgecolor=DARK_BG, linewidth=0.5)
    ax6.add_patch(box)
    ax6.text(0.50, cy + 0.055, title, ha="center", va="center",
             fontsize=8.5, fontweight="bold", color=fg)
    ax6.text(0.50, cy + 0.012, sub,  ha="center", va="center",
             fontsize=7.2, color=TEXT_DIM if bg != ACCENT else "#633806")

for i in range(len(steps)-1):
    y_top = steps[i][2] - 0.08
    y_bot = steps[i+1][2] + 0.075
    ax6.annotate("", xy=(0.5, y_bot), xytext=(0.5, y_top),
                 arrowprops=dict(arrowstyle="->", color=TEXT_DIM,
                                 lw=1.2))

ax6.text(0.50, 0.975, "Pipeline", ha="center", va="top",
         fontsize=9, fontweight="bold", color=TEXT_MAIN)

# ════════════════════════════════════════════════════════════
#  PANEL 7 — Metric stat cards  (row 2, col 0–1)
# ════════════════════════════════════════════════════════════
ax7 = fig.add_subplot(gs[2, :2])
ax7.set_facecolor(DARK_BG)
ax7.set_xlim(0, 1)
ax7.set_ylim(0, 1)
ax7.axis("off")

stats = [
    ("Accuracy",   f"{acc*100:.1f}%",  ACCENT2),
    ("F1 Score",   f"{f1:.3f}",        ACCENT),
    ("Train size", "120",              TEXT_DIM),
    ("Test size",  "30",               TEXT_DIM),
    ("Classes",    "3",                TEXT_DIM),
    ("Features",   "4",                TEXT_DIM),
]

n    = len(stats)
w    = 0.90 / n
pad  = 0.017

for i, (lbl, val, col) in enumerate(stats):
    x0 = 0.05 + i * (w + pad)
    box = FancyBboxPatch((x0, 0.12), w, 0.76,
                         boxstyle="round,pad=0.015",
                         facecolor=MID_BG, edgecolor=CARD_BG, linewidth=0.5)
    ax7.add_patch(box)
    ax7.text(x0 + w/2, 0.62, val,
             ha="center", va="center",
             fontsize=16, fontweight="bold", color=col)
    ax7.text(x0 + w/2, 0.27, lbl,
             ha="center", va="center",
             fontsize=7.5, color=TEXT_DIM)

# ════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════
fig.text(0.97, 0.018,
         "#MachineLearning  #Python  #DataScience  #KNN  #SupervisedLearning",
         ha="right", va="bottom",
         fontsize=8, color=CARD_BG)

fig.text(0.03, 0.018,
         "DecodeLabs · Project 2 · Iris KNN Classifier",
         ha="left", va="bottom",
         fontsize=8, color=CARD_BG)

# ── Save ─────────────────────────────────────────────────────
out = "linkedin_project2.png"
plt.savefig(out, dpi=150, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
plt.show()
print(f"\nSaved → {out}  (ready to upload on LinkedIn)")