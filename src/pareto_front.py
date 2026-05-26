import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# (label, mAP50-95, IPS, Total Power mW)
models = [
    ("Base FP32 1280",  0.761,  28.05,  9892),
    ("KD FP32 1280",    0.855,  27.97,  9939),
    ("Base INT8 1280",  0.778,  32.23,  5161),
    ("KD INT8 1280",    0.854,  31.93,  5001),
    ("Base FP32 640",   0.743,  68.83,  7644),
    ("KD FP32 640",     0.859,  69.10,  7523),
    ("Base INT8 640",   0.742, 105.74,  5854),
    ("KD INT8 640",     0.862, 105.95,  5837),
]

labels     = [m[0] for m in models]
map_vals   = [m[1] for m in models]
ips_vals   = [m[2] for m in models]
power_vals = [m[3] for m in models]

is_kd      = ["KD" in l for l in labels]
is_optimal = [l == "KD INT8 640" for l in labels]

kd_color   = "#d95f02"
base_color = "#1b9e77"
opt_color  = "#FFD700"   # gold — visually distinct for the best point

# Bubble area proportional to power — fixed reference so ratios are honest
min_p, max_p = min(power_vals), max(power_vals)
sizes = [p / 10000 * 500 for p in power_vals]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

# Draw Base bubbles
for i in range(len(labels)):
    if not is_kd[i]:
        ax.scatter(ips_vals[i], map_vals[i],
                   s=sizes[i], c=base_color, marker="o",
                   alpha=0.70, edgecolors="black", linewidths=1.2, zorder=3)

# Draw KD bubbles — except the optimal one
for i in range(len(labels)):
    if is_kd[i] and not is_optimal[i]:
        ax.scatter(ips_vals[i], map_vals[i],
                   s=sizes[i], c=kd_color, marker="o",
                   alpha=0.70, edgecolors="black", linewidths=1.2, zorder=3)

# Draw KD INT8 640 as a gold circle — clearly the best point
kd_opt_idx = labels.index("KD INT8 640")
ax.scatter(ips_vals[kd_opt_idx], map_vals[kd_opt_idx],
           s=sizes[kd_opt_idx], c=opt_color, marker="o",
           edgecolors="black", linewidths=1.8, zorder=6)

# Labels with offsets to avoid overlap
offsets = {
    "Base FP32 1280": (-72, -18),
    "KD FP32 1280":   (-70,  10),
    "Base INT8 1280": ( 10,  -18),
    "KD INT8 1280":   ( 10,   10),
    "Base FP32 640":  (  8,  -18),
    "KD FP32 640":    (  8,   10),
    "Base INT8 640":  (  8,  -18),
    "KD INT8 640":    ( 10,   12),
}

for i, label in enumerate(labels):
    ox, oy = offsets.get(label, (8, 5))
    weight = "bold" if is_optimal[i] else ("semibold" if is_kd[i] else "normal")
    ax.annotate(label, (ips_vals[i], map_vals[i]),
                textcoords="offset points", xytext=(ox, oy),
                fontsize=8.5, fontweight=weight)

# "Pareto optimal" annotation with arrow for the gold star
ax.annotate("Pareto-optimal", xy=(ips_vals[kd_opt_idx], map_vals[kd_opt_idx]),
            xytext=(ips_vals[kd_opt_idx] - 38, map_vals[kd_opt_idx] + 0.015),
            fontsize=9, color="#8B6914", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#8B6914", lw=1.3))

# ── Base: Pareto front line (genuine trade-off: max mAP vs max IPS) ───────────
# Two-point front: Base INT8 1280 (max mAP) ↔ Base INT8 640 (max IPS).
# Base FP32 640 has only +0.001 mAP over Base INT8 640 at -54% IPS — treated
# as effectively dominated (within measurement noise).
MAP_TOL = 0.005
base_idx = [i for i in range(len(labels)) if not is_kd[i]]
pareto_b = []
for i in base_idx:
    dominated = any(
        ips_vals[j] >= ips_vals[i] and map_vals[j] >= map_vals[i] - MAP_TOL and
        (ips_vals[j] > ips_vals[i] or map_vals[j] > map_vals[i])
        for j in base_idx if j != i
    )
    if not dominated:
        pareto_b.append((ips_vals[i], map_vals[i]))

pareto_b = sorted(pareto_b, key=lambda x: x[0])
pareto_ips_b, pareto_map_b = zip(*pareto_b)
ax.step(pareto_ips_b, pareto_map_b, where="post",
        color="#1b9e77", linewidth=1.8, linestyle="--",
        alpha=0.85, zorder=2)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel("Median IPS   (higher → better)", fontsize=12)
ax.set_ylabel("mAP50-95   (higher → better)", fontsize=12)
ax.set_title("Pareto Front: Accuracy vs Throughput vs Power Draw",
             fontsize=13, fontweight="bold", pad=14)

ax.set_xlim(0, 125)
ax.set_ylim(0.68, 0.90)
ax.grid(True, linestyle="--", alpha=0.4)

# Arrow annotations showing desired direction
ax.annotate("", xy=(120, 0.686), xytext=(108, 0.686),
            arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
ax.annotate("", xy=(2, 0.895), xytext=(2, 0.882),
            arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))

# ── Legend: colour ────────────────────────────────────────────────────────────
legend_color = [
    mpatches.Patch(facecolor="#d95f02", edgecolor="black", label="KD student (M2)"),
    mpatches.Patch(facecolor="#1b9e77", edgecolor="black", label="Base model (M0)"),
]

# ── Legend: bubble size ───────────────────────────────────────────────────────
bubble_legend = [
    ax.scatter([], [], s=p / 10000 * 500,
               c="grey", alpha=0.6, edgecolors="black", label=f"{p:,} mW")
    for p in [5000, 7500, 10000]
]

pareto_lines = [
    plt.scatter([], [], marker="o", s=160, c=opt_color, edgecolors="black",
                label="KD Pareto-optimal point"),
    plt.Line2D([0], [0], color="#1b9e77", lw=1.8, linestyle="--",
               label="Base Pareto front"),
]

first_legend = ax.legend(handles=legend_color + pareto_lines, loc="lower right",
                          fontsize=9, title="Model type", title_fontsize=9)
ax.add_artist(first_legend)
ax.legend(handles=bubble_legend, loc="center right",
          fontsize=9, title="Total power\n(bubble size)", title_fontsize=9)

plt.tight_layout()
plt.savefig("pareto_front.pdf", bbox_inches="tight", dpi=300)
plt.savefig("pareto_front.png", bbox_inches="tight", dpi=300)
print("Saved: pareto_front.pdf  /  pareto_front.png")
plt.show()
