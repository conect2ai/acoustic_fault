"""
Embedded Benchmark Figures for TinyML Acoustic Fault Detection Paper
Generates Figures 4–8 for Section 6.2 (Embedded Deployment Benchmark)

Data: resultado_deploys_ALL.csv
Paper standard: ACM TECS
"""

import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import LogLocator, LogFormatter
import warnings
warnings.filterwarnings("ignore")

# ── Reproduce the CSV inline ────────────────────────────────────────────────
RAW = """model_name,feature_type,Parameters,Sketch,Global,Inference_ms,Current_mA,Voltage_V,Power_mW,MCU,Accuracy
CNN,MFCC,16012,0,0,0,0,5.014,0,ESP8266,0.9651162791
CNN,MFCC,16012,0,0,0,0,5.014,0,STM32F103C,0.9651162791
CNN,MFCC,16012,346344,63028,9.41,61.1,5.014,306.3554,ESP32,0.9651162791
CNN,MFCC,16012,160482,83784,162.07,25.15,5.014,126.1021,Raspberry Pi Pico,0.9651162791
CNN,MFCC,16012,143748,110480,67.41,33.02,5.014,165.56228,Raspberry Pi Pico W,0.9651162791
CNN,MFCC,16012,139392,110908,10.73,14.6,5.014,73.2044,Raspberry Pi Pico 2 W,0.9651162791
CNN,MFCC,16012,116868,50672,10.69,16.6,5.014,83.2324,Raspberry Pi Pico 2,0.9651162791
CNN,MFCC,16012,151496,85408,24.06,18.07,5.014,90.60298,Arduino Nano 33 BLE SENSE,0.9651162791
CNN,MFCC,16012,201504,103944,2.56,111.1,5.014,557.0554,Arduino Portenta H7,0.9651162791
CNN,MFE,16012,0,0,0,0,5.014,0,ESP8266,0.9825581395
CNN,MFE,16012,0,0,0,0,5.014,0,STM32F103C,0.9825581395
CNN,MFE,16012,346344,63028,9.41,61.1,5.014,306.3554,ESP32,0.9825581395
CNN,MFE,16012,160482,83784,162.07,25.15,5.014,126.1021,Raspberry Pi Pico,0.9825581395
CNN,MFE,16012,143748,110480,67.41,33.02,5.014,165.56228,Raspberry Pi Pico W,0.9825581395
CNN,MFE,16012,139392,110908,10.73,14.6,5.014,73.2044,Raspberry Pi Pico 2 W,0.9825581395
CNN,MFE,16012,116868,50672,10.69,16.6,5.014,83.2324,Raspberry Pi Pico 2,0.9825581395
CNN,MFE,16012,151496,85408,24.06,18.07,5.014,90.60298,Arduino Nano 33 BLE SENSE,0.9825581395
CNN,MFE,16012,201504,103944,2.56,111.1,5.014,557.0554,Arduino Portenta H7,0.9825581395
FAN,MFCC,7984,309504,62632,15.4,19.1,3.3,63.03,ESP8266,0.988372093
FAN,MFCC,7984,55432,2912,17.16,9.9,5.014,49.6386,STM32F103C,0.988372093
FAN,MFCC,7984,317376,23556,0.67,53.39,5.014,267.69746,ESP32,0.988372093
FAN,MFCC,7984,132305,44296,20.7,25.13,5.014,126.00182,Raspberry Pi Pico,0.988372093
FAN,MFCC,7984,111444,70992,8.73,31.83,5.014,159.59562,Raspberry Pi Pico W,0.988372093
FAN,MFCC,7984,107320,71428,2.03,14.51,5.014,72.75314,Raspberry Pi Pico 2 W,0.988372093
FAN,MFCC,7984,84652,10712,2.04,16.25,5.014,81.4775,Raspberry Pi Pico 2,0.988372093
FAN,MFCC,7984,123720,46032,1.47,26.85,5.014,134.6259,Arduino Nano 33 BLE SENSE,0.988372093
FAN,MFCC,7984,173112,63984,0.2,111.10,5.014,557.0554,Arduino Portenta H7,0.988372093
FAN,MFE,7792,310096,61384,14.26,21.32,3.3,70.356,ESP8266,0.9860465116
FAN,MFE,7792,55592,2432,15.57,10.03,5.014,50.29042,STM32F103C,0.9860465116
FAN,MFE,7792,317764,23076,0.85,50.68,5.014,254.10952,ESP32,0.9860465116
FAN,MFE,7792,132491,43816,19.6,25.43,5.014,127.50602,Raspberry Pi Pico,0.9860465116
FAN,MFE,7792,110764,70512,8.53,32.3,5.014,161.9522,Raspberry Pi Pico W,0.9860465116
FAN,MFE,7792,106976,70948,1.96,14.8,5.014,74.2072,Raspberry Pi Pico 2 W,0.9860465116
FAN,MFE,7792,84316,10232,1.95,16.1,5.014,80.7254,Raspberry Pi Pico 2,0.9860465116
FAN,MFE,7792,123784,45552,1.38,18,5.014,90.252,Arduino Nano 33 BLE SENSE,0.9860465116
FAN,MFE,7792,173176,63504,0.21,110.7,5.014,555.0498,Arduino Portenta H7,0.9860465116
MLP,MFCC,2140,284192,37304,3.9,16.9,3.3,55.77,ESP8266,0.9604651163
MLP,MFCC,2140,28868,2040,5.11,9.8,5.014,49.1372,STM32F103C,0.9604651163
MLP,MFCC,2140,290696,22548,0.08,50.9,5.014,255.2126,ESP32,0.9604651163
MLP,MFCC,2140,104866,43304,4.24,25.4,5.014,127.3556,Raspberry Pi Pico,0.9604651163
MLP,MFCC,2140,87532,70000,1.76,27.2,5.014,136.3808,Raspberry Pi Pico W,0.9604651163
MLP,MFCC,2140,83280,70428,0.15,14.8,5.014,74.2072,Raspberry Pi Pico 2 W,0.9604651163
MLP,MFCC,2140,60668,9716,0.14,15.3,5.014,76.7142,Raspberry Pi Pico 2,0.9604651163
MLP,MFCC,2140,96864,45032,0.75,20.3,5.014,101.7842,Arduino Nano 33 BLE SENSE,0.9604651163
MLP,MFCC,2140,145904,62992,0.04,109.10,5.014,547.0274,Arduino Portenta H7,0.9604651163
MLP,MFE,2140,284192,37304,3.9,16.9,3.3,55.77,ESP8266,0.9302325581
MLP,MFE,2140,28868,2040,5.11,9.8,5.014,49.1372,STM32F103C,0.9302325581
MLP,MFE,2140,290696,22548,0.08,50.9,5.014,255.2126,ESP32,0.9302325581
MLP,MFE,2140,104866,43304,4.24,25.4,5.014,127.3556,Raspberry Pi Pico,0.9302325581
MLP,MFE,2140,87532,70000,1.76,27.2,5.014,136.3808,Raspberry Pi Pico W,0.9302325581
MLP,MFE,2140,83280,70428,0.15,14.8,5.014,74.2072,Raspberry Pi Pico 2 W,0.9302325581
MLP,MFE,2140,60668,9716,0.14,15.3,5.014,76.7142,Raspberry Pi Pico 2,0.9302325581
MLP,MFE,2140,96864,45032,0.75,20.3,5.014,101.7842,Arduino Nano 33 BLE SENSE,0.9302325581
MLP,MFE,2140,145904,62992,0.04,109.10,5.014,547.0274,Arduino Portenta H7,0.9302325581
KAN,MFCC,19243,0,0,0,0,3.3,0,ESP8266,0.8848837209
KAN,MFCC,19243,0,0,0,0,5.014,0,STM32F103C,0.8848837209
KAN,MFCC,19243,818168,22548,87.19,68.1,5.014,341.4534,ESP32,0.8848837209
KAN,MFCC,19243,584553,43304,341.87,26.6,5.014,133.3724,Raspberry Pi Pico,0.8848837209
KAN,MFCC,19243,563708,70048,83.76,36,5.014,180.504,Raspberry Pi Pico W,0.8848837209
KAN,MFCC,19243,618040,70428,56.08,18.8,5.014,94.2632,Raspberry Pi Pico 2 W,0.8848837209
KAN,MFCC,19243,595380,9712,55.69,21.3,5.014,106.7982,Raspberry Pi Pico 2,0.8848837209
KAN,MFCC,19243,636424,45032,202.18,21.2,5.014,106.2968,Arduino Nano 33 BLE SENSE,0.8848837209
KAN,MFCC,19243,526264,62984,3.4,120.00,5.014,601.68,Arduino Portenta H7,0.8848837209
KAN,MFE,23317,0,0,0,0,3.3,0,ESP8266,0.8813953488
KAN,MFE,23317,0,0,0,0,5.014,0,STM32F103C,0.8813953488
KAN,MFE,23317,1003612,22548,121.89,68.7,5.014,344.4618,ESP32,0.8813953488
KAN,MFE,23317,756965,43304,487.21,26.7,5.014,133.8738,Raspberry Pi Pico,0.8813953488
KAN,MFE,23317,736052,70048,114.99,35.5,5.014,177.997,Raspberry Pi Pico W,0.8813953488
KAN,MFE,23317,796320,70428,72.31,18.7,5.014,93.7618,Raspberry Pi Pico 2 W,0.8813953488
KAN,MFE,23317,773660,9712,71.71,20.3,5.014,101.7842,Raspberry Pi Pico 2,0.8813953488
KAN,MFE,23317,814600,45032,286.72,21.4,5.014,107.2996,Arduino Nano 33 BLE SENSE,0.8813953488
KAN,MFE,23317,646264,62984,4.5,124.20,5.014,622.7388,Arduino Portenta H7,0.8813953488
RBFN,MFCC,26876,0,0,0,0,3.3,0,ESP8266,0.973255814
RBFN,MFCC,26876,0,0,0,0,5.014,0,STM32F103C,0.973255814
RBFN,MFCC,26876,389872,23348,8.75,57.2,5.014,286.8008,ESP32,0.973255814
RBFN,MFCC,26876,204117,44108,99.6,26,5.014,130.364,Raspberry Pi Pico,0.973255814
RBFN,MFCC,26876,186656,70800,39.46,32.6,5.014,163.4564,Raspberry Pi Pico W,0.973255814
RBFN,MFCC,26876,182364,71228,7.36,16.9,5.014,84.7366,Raspberry Pi Pico 2 W,0.973255814
RBFN,MFCC,26876,159696,10512,7.32,17.6,5.014,88.2464,Raspberry Pi Pico 2,0.973255814
RBFN,MFCC,26876,195992,45832,6.22,18.4,5.014,92.2576,Arduino Nano 33 BLE SENSE,0.973255814
RBFN,MFCC,26876,245136,63792,0.79,114.90,5.014,576.1086,Arduino Portenta H7,0.973255814
RBFN,MFE,26876,0,0,0,0,3.3,0,ESP8266,0.9697674419
RBFN,MFE,26876,0,0,0,0,5.014,0,STM32F103C,0.9697674419
RBFN,MFE,26876,389872,23348,8.75,57.2,5.014,286.8008,ESP32,0.9697674419
RBFN,MFE,26876,204117,44108,99.6,26,5.014,130.364,Raspberry Pi Pico,0.9697674419
RBFN,MFE,26876,186656,70800,39.46,32.6,5.014,163.4564,Raspberry Pi Pico W,0.9697674419
RBFN,MFE,26876,182364,71228,7.36,16.9,5.014,84.7366,Raspberry Pi Pico 2 W,0.9697674419
RBFN,MFE,26876,159696,10512,7.32,17.6,5.014,88.2464,Raspberry Pi Pico 2,0.9697674419
RBFN,MFE,26876,195992,45832,6.22,18.4,5.014,92.2576,Arduino Nano 33 BLE SENSE,0.9697674419
RBFN,MFE,26876,245136,63792,0.79,114.90,5.014,576.1086,Arduino Portenta H7,0.9697674419
"""

import pandas as pd

df = pd.read_csv(io.StringIO(RAW.strip()))

# Filter only deployable entries (Inference_ms > 0)
df_dep = df[df["Inference_ms"] > 0].copy()

# Compute energy per inference in µJ  (mW × ms = µJ)
df_dep["Energy_uJ"] = df_dep["Power_mW"] * df_dep["Inference_ms"]

# ── Styling constants ────────────────────────────────────────────────────────
MODEL_COLORS = {
    "FAN":  "#1a6fb5",   # strong blue
    "CNN":  "#e07b00",   # amber
    "MLP":  "#2ca02c",   # green
    "RBFN": "#9467bd",   # purple
    "KAN":  "#d62728",   # red
}
FEAT_MARKERS = {"MFCC": "o", "MFE": "s"}
FEAT_LS      = {"MFCC": "-",  "MFE": "--"}

# FPU presence flag for each MCU
FPU_MAP = {
    "ESP8266":              False,
    "STM32F103C":           False,
    "ESP32":                True,
    "Raspberry Pi Pico":    False,
    "Raspberry Pi Pico W":  False,
    "Raspberry Pi Pico 2":  True,
    "Raspberry Pi Pico 2 W":True,
    "Arduino Nano 33 BLE SENSE": True,
    "Arduino Portenta H7":  True,
}

# Ordered MCU list for bar charts
MCU_ORDER = [
    "ESP8266",
    "STM32F103C",
    "Raspberry Pi Pico",
    "Raspberry Pi Pico W",
    "ESP32",
    "Raspberry Pi Pico 2 W",
    "Raspberry Pi Pico 2",
    "Arduino Nano 33 BLE SENSE",
    "Arduino Portenta H7",
]
MCU_LABELS = {
    "ESP8266":              "ESP8266\n80 MHz\nno FPU",
    "STM32F103C":           "STM32F103C\n72 MHz\nno FPU",
    "Raspberry Pi Pico":    "RPico\n133 MHz\nno FPU",
    "Raspberry Pi Pico W":  "RPico W\n133 MHz\nno FPU",
    "ESP32":                "ESP32\n240 MHz\nFPU",
    "Raspberry Pi Pico 2 W":"RPico 2 W\n150 MHz\nFPU",
    "Raspberry Pi Pico 2":  "RPico 2\n150 MHz\nFPU",
    "Arduino Nano 33 BLE SENSE": "Nano33\n64 MHz\nFPU",
    "Arduino Portenta H7":  "PortentaH7\n480 MHz\nFPU",
}

MODEL_ORDER = ["FAN", "CNN", "MLP", "RBFN", "KAN"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":    True,
    "grid.alpha":   0.35,
    "grid.linestyle": "--",
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – Inference Latency (ms) per Model × MCU  [grouped bar, log scale]
# ─────────────────────────────────────────────────────────────────────────────
def fig4_latency_per_mcu():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    features = ["MFCC", "MFE"]

    for ax, feat in zip(axes, features):
        subset = df_dep[df_dep["feature_type"] == feat]
        n_mcu   = len(MCU_ORDER)
        n_model = len(MODEL_ORDER)
        width   = 0.14
        x       = np.arange(n_mcu)

        for i, model in enumerate(MODEL_ORDER):
            vals = []
            for mcu in MCU_ORDER:
                row = subset[(subset["model_name"] == model) & (subset["MCU"] == mcu)]
                vals.append(row["Inference_ms"].values[0] if len(row) > 0 else np.nan)

            offset = (i - n_model / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width=width * 0.9,
                          color=MODEL_COLORS[model], label=model,
                          zorder=3, edgecolor="white", linewidth=0.4)

        # FPU divider
        ax.axvline(x=3.5, color="black", linewidth=1.2, linestyle=":", alpha=0.7)
        ax.text(1.5, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 600,
                "no FPU", ha="center", va="center", fontsize=7.5,
                color="gray", style="italic")
        ax.text(5.5, 0.05, "FPU", ha="center", va="bottom",
                fontsize=7.5, color="gray", style="italic")

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([MCU_LABELS[m] for m in MCU_ORDER], fontsize=6.5)
        ax.set_title(f"Feature: {feat}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Inference latency (ms, log scale)" if feat == "MFCC" else "")
        ax.set_xlabel("Target microcontroller")
        ax.yaxis.grid(True, which="both", alpha=0.3)

        # Annotate non-deployable with ✕
        for i, mcu in enumerate(MCU_ORDER):
            for j, model in enumerate(MODEL_ORDER):
                row = subset[(subset["model_name"] == model) & (subset["MCU"] == mcu)]
                if len(row) == 0 or row["Inference_ms"].values[0] == 0:
                    if mcu in ["ESP8266", "STM32F103C"] and model not in ["FAN", "MLP"]:
                        offset = (j - n_model / 2 + 0.5) * width
                        ax.text(i + offset, 0.12, "✕", ha="center", va="bottom",
                                fontsize=6, color="#cc0000", fontweight="bold")

    axes[0].legend(loc="upper right", ncol=1, framealpha=0.85,
                   title="Architecture", title_fontsize=8)

    # Restore ylim after annotations
    for ax in axes:
        ax.set_ylim(bottom=0.03)

    fig.suptitle("Figure 4 — Inference Latency per Architecture and Target MCU",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig4_latency_per_mcu.pdf",
                bbox_inches="tight", format="pdf")
    fig.savefig("/mnt/user-data/outputs/fig4_latency_per_mcu.png",
                bbox_inches="tight", dpi=200)
    print("  ✓  Figure 4 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – Energy per Inference (µJ) per Model × MCU  [grouped bar, log scale]
# ─────────────────────────────────────────────────────────────────────────────
def fig5_energy_per_mcu():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    features = ["MFCC", "MFE"]

    for ax, feat in zip(axes, features):
        subset = df_dep[df_dep["feature_type"] == feat]
        n_mcu   = len(MCU_ORDER)
        n_model = len(MODEL_ORDER)
        width   = 0.14
        x       = np.arange(n_mcu)

        for i, model in enumerate(MODEL_ORDER):
            vals = []
            for mcu in MCU_ORDER:
                row = subset[(subset["model_name"] == model) & (subset["MCU"] == mcu)]
                vals.append(row["Energy_uJ"].values[0] if len(row) > 0 else np.nan)

            offset = (i - n_model / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width * 0.9,
                   color=MODEL_COLORS[model], label=model,
                   zorder=3, edgecolor="white", linewidth=0.4)

        ax.axvline(x=3.5, color="black", linewidth=1.2, linestyle=":", alpha=0.7)
        ax.text(1.5, 200000, "no FPU", ha="center", va="center",
                fontsize=7.5, color="gray", style="italic")
        ax.text(6.0, 200000, "FPU", ha="center", va="center",
                fontsize=7.5, color="gray", style="italic")

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([MCU_LABELS[m] for m in MCU_ORDER], fontsize=6.5)
        ax.set_title(f"Feature: {feat}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Energy per inference (µJ, log scale)" if feat == "MFCC" else "")
        ax.set_xlabel("Target microcontroller")
        ax.yaxis.grid(True, which="both", alpha=0.3)

    axes[0].legend(loc="upper right", ncol=1, framealpha=0.85,
                   title="Architecture", title_fontsize=8)
    axes[0].set_ylim(bottom=1)

    fig.suptitle("Figure 5 — Energy per Inference Event per Architecture and Target MCU",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig5_energy_per_mcu.pdf",
                bbox_inches="tight", format="pdf")
    fig.savefig("/mnt/user-data/outputs/fig5_energy_per_mcu.png",
                bbox_inches="tight", dpi=200)
    print("  ✓  Figure 5 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 – Accuracy vs Energy per Inference – Pareto frontier
# ─────────────────────────────────────────────────────────────────────────────
def pareto_front(x_vals, y_vals):
    """Return boolean mask of non-dominated points (min x, max y)."""
    n   = len(x_vals)
    dom = np.zeros(n, dtype=bool)
    for i in range(n):
        dominated = False
        for j in range(n):
            if j == i: continue
            if x_vals[j] <= x_vals[i] and y_vals[j] >= y_vals[i]:
                if x_vals[j] < x_vals[i] or y_vals[j] > y_vals[i]:
                    dominated = True
                    break
        dom[i] = not dominated
    return dom


def fig6_pareto_acc_energy():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Use best feature per model for each MCU (highest accuracy = MFCC for most)
    # Plot all deployable points
    for model in MODEL_ORDER:
        for feat in ["MFCC", "MFE"]:
            sub = df_dep[(df_dep["model_name"] == model) &
                         (df_dep["feature_type"] == feat)]
            if sub.empty: continue
            ax.scatter(sub["Energy_uJ"], sub["Accuracy"] * 100,
                       color=MODEL_COLORS[model],
                       marker=FEAT_MARKERS[feat],
                       s=55, alpha=0.72, zorder=4,
                       edgecolors="white", linewidths=0.5)

    # Compute and draw Pareto frontier using best-of-feature per MCU
    # For each (model, MCU), take MFCC (highest accuracy)
    best = df_dep.loc[df_dep.groupby(["model_name", "MCU"])["Accuracy"].idxmax()]
    xe = best["Energy_uJ"].values
    ya = best["Accuracy"].values * 100
    mask = pareto_front(xe, ya)
    px = xe[mask]
    py = ya[mask]
    # Sort by energy for line
    sort_idx = np.argsort(px)
    px, py = px[sort_idx], py[sort_idx]
    ax.plot(px, py, color="black", linewidth=1.5,
            linestyle="--", zorder=5, label="Pareto frontier", alpha=0.8)
    ax.scatter(px, py, color="black", s=80, zorder=6,
               marker="D", alpha=0.9, edgecolors="black")

    # Annotate key Pareto points
    annotations = {
        ("FAN",  "Raspberry Pi Pico 2"):    ("FAN\nRPico2\n98.8%",   "left"),
        ("FAN",  "Arduino Portenta H7"):     ("FAN\nPortentaH7",       "right"),
        ("MLP",  "Raspberry Pi Pico 2"):     ("MLP\nRPico2\n96.0%",   "left"),
        ("MLP",  "Arduino Portenta H7"):     ("MLP\nPortentaH7",       "right"),
    }
    for (m, mcu), (label, ha) in annotations.items():
        row = df_dep[(df_dep["model_name"] == m) & (df_dep["MCU"] == mcu) &
                     (df_dep["feature_type"] == "MFCC")]
        if row.empty: continue
        xe_ = row["Energy_uJ"].values[0]
        ya_ = row["Accuracy"].values[0] * 100
        ax.annotate(label, xy=(xe_, ya_),
                    xytext=(xe_ * (1.8 if ha == "right" else 0.55), ya_ - 0.5),
                    fontsize=7, ha=ha, color=MODEL_COLORS[m],
                    arrowprops=dict(arrowstyle="-", color=MODEL_COLORS[m],
                                   lw=0.8, alpha=0.7))

    ax.set_xscale("log")
    ax.set_xlabel("Energy per inference (µJ, log scale)", fontsize=10)
    ax.set_ylabel("Test accuracy (%)", fontsize=10)
    ax.set_xlim(left=5)
    ax.set_ylim([84, 100.2])

    # Legend for models
    model_patches = [mpatches.Patch(color=MODEL_COLORS[m], label=m)
                     for m in MODEL_ORDER]
    feat_handles  = [mlines.Line2D([], [], color="gray", marker=FEAT_MARKERS[f],
                                   linestyle="None", markersize=7, label=f)
                     for f in ["MFCC", "MFE"]]
    pareto_handle = mlines.Line2D([], [], color="black", linestyle="--",
                                  marker="D", markersize=6, label="Pareto front")
    ax.legend(handles=model_patches + feat_handles + [pareto_handle],
              loc="lower right", ncol=2, fontsize=8, framealpha=0.9)

    ax.set_title("Figure 6 — Embedded Pareto Frontier: Test Accuracy vs. Energy per Inference",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig6_pareto_acc_energy.pdf",
                bbox_inches="tight", format="pdf")
    fig.savefig("/mnt/user-data/outputs/fig6_pareto_acc_energy.png",
                bbox_inches="tight", dpi=200)
    print("  ✓  Figure 6 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 – Inference Time vs Number of Parameters (FPU vs no-FPU annotated)
# ─────────────────────────────────────────────────────────────────────────────
def fig7_latency_vs_params():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, feat in zip(axes, ["MFCC", "MFE"]):
        sub = df_dep[df_dep["feature_type"] == feat].copy()
        sub["has_fpu"] = sub["MCU"].map(FPU_MAP)

        for model in MODEL_ORDER:
            ms = sub[sub["model_name"] == model]
            for _, row in ms.iterrows():
                marker = "^" if row["has_fpu"] else "v"
                ax.scatter(row["Parameters"], row["Inference_ms"],
                           color=MODEL_COLORS[model],
                           marker=marker, s=70, alpha=0.8,
                           edgecolors="white", linewidths=0.5, zorder=4)

        # Trend lines per model (log–log regression)
        for model in MODEL_ORDER:
            ms = sub[sub["model_name"] == model]
            if len(ms) < 2: continue
            ax.scatter([], [], color=MODEL_COLORS[model],
                       label=model, s=55, zorder=5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of parameters (log scale)", fontsize=9)
        ax.set_ylabel("Inference time (ms, log scale)" if feat == "MFCC" else "")
        ax.set_title(f"Feature: {feat}", fontsize=10, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)

    # Shared legend
    model_patches = [mpatches.Patch(color=MODEL_COLORS[m], label=m)
                     for m in MODEL_ORDER]
    fpu_handles = [
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None",
                      markersize=8, label="FPU present"),
        mlines.Line2D([], [], color="gray", marker="v", linestyle="None",
                      markersize=8, label="No FPU"),
    ]
    axes[1].legend(handles=model_patches + fpu_handles,
                   loc="upper left", ncol=2, fontsize=8, framealpha=0.9)

    fig.suptitle("Figure 7 — Inference Time vs. Parameter Count "
                 "(▲ FPU-equipped  ▼ software-float)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig7_latency_vs_params.pdf",
                bbox_inches="tight", format="pdf")
    fig.savefig("/mnt/user-data/outputs/fig7_latency_vs_params.png",
                bbox_inches="tight", dpi=200)
    print("  ✓  Figure 7 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 – Energy per Inference vs Latency (iso-energy + architecture colour)
# ─────────────────────────────────────────────────────────────────────────────
def _iso_label_pos(E_uJ, T_min=0.03, T_max=700, P_min=30, P_max=800):
    """
    Return (T_label, P_label) for an iso-energy contour E = P * T.
    Places the label at a fixed fractional position along the visible segment
    of the contour (inside xlim × ylim), biased toward the upper-left corner
    so it stays away from data markers.
    Returns None if the contour is not visible at all.
    """
    # Visible T range: intersection of [T_min, T_max] and [E/P_max, E/P_min]
    T_lo = max(T_min, E_uJ / P_max)
    T_hi = min(T_max, E_uJ / P_min)
    if T_lo >= T_hi:
        return None
    # Place label at 25 % of the log-range (upper-left portion)
    T_label = np.exp(np.log(T_lo) + 0.25 * (np.log(T_hi) - np.log(T_lo)))
    P_label = E_uJ / T_label
    return T_label, P_label


def fig8_energy_vs_latency():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # Axis limits – defined here so label placement uses the same values
    X_LIM = (0.03, 700)
    Y_LIM = (30,   800)

    # iso-energy lines (µJ)
    iso_vals = [50, 200, 1000, 10000, 100000]
    t_range  = np.logspace(np.log10(X_LIM[0]), np.log10(X_LIM[1]), 500)

    for ax, feat in zip(axes, ["MFCC", "MFE"]):
        sub = df_dep[df_dep["feature_type"] == feat].copy()

        # draw iso-energy contours clipped to the visible axes region
        for E in iso_vals:
            P_line = E / t_range          # P = E / T  → mW
            # Only keep segments within ylim so matplotlib doesn't
            # auto-extend the axes when tight_layout runs
            mask   = (P_line >= Y_LIM[0]) & (P_line <= Y_LIM[1])
            if mask.any():
                ax.plot(t_range[mask], P_line[mask], color="lightgray",
                        linewidth=0.9, linestyle="-", zorder=1, clip_on=True)

            # Label positioned along the visible part of the contour
            pos = _iso_label_pos(E, T_min=X_LIM[0], T_max=X_LIM[1],
                                    P_min=Y_LIM[0], P_max=Y_LIM[1])
            if pos is not None:
                T_lbl, P_lbl = pos
                # Rotation: in log-log space the slope is -1 (45°).
                # Correct visual angle depends on axes aspect ratio;
                # -38° is a good approximation for a 2:1 figure width.
                ax.text(T_lbl, P_lbl * 1.18,
                        f"{E:,} µJ".replace(",", "\u202f"),
                        fontsize=6.5, color="#888888",
                        ha="center", va="bottom",
                        rotation=-38, rotation_mode="anchor",
                        clip_on=True, zorder=2)

        for model in MODEL_ORDER:
            ms = sub[sub["model_name"] == model]
            ax.scatter(ms["Inference_ms"], ms["Power_mW"],
                       color=MODEL_COLORS[model],
                       marker=FEAT_MARKERS[feat],
                       s=65, alpha=0.85, zorder=4,
                       edgecolors="white", linewidths=0.5, label=model)

        # Annotate FAN and MLP on key MCUs
        for model, mcu_label in [("FAN", "Raspberry Pi Pico 2"),
                                  ("FAN", "Raspberry Pi Pico"),
                                  ("KAN", "Raspberry Pi Pico")]:
            row = sub[(sub["model_name"] == model) & (sub["MCU"] == mcu_label)]
            if row.empty: continue
            r = row.iloc[0]
            ax.annotate(f"{model}\n{mcu_label.replace('Raspberry Pi ', 'R')}",
                        xy=(r["Inference_ms"], r["Power_mW"]),
                        xytext=(r["Inference_ms"] * 2.0, r["Power_mW"] * 1.6),
                        fontsize=6.5, color=MODEL_COLORS[model],
                        arrowprops=dict(arrowstyle="-", color=MODEL_COLORS[model],
                                        lw=0.7, alpha=0.7))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Inference latency (ms, log scale)", fontsize=9)
        ax.set_ylabel("Instantaneous power draw (mW, log scale)"
                      if feat == "MFCC" else "")
        ax.set_title(f"Feature: {feat}", fontsize=10, fontweight="bold")
        ax.set_xlim(X_LIM)
        ax.set_ylim(Y_LIM)
        ax.grid(True, which="both", alpha=0.25)

    # Build legend (deduplicate)
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    axes[1].legend(seen.values(), seen.keys(),
                   loc="upper right", ncol=1, fontsize=8, framealpha=0.9,
                   title="Architecture", title_fontsize=8)

    fig.suptitle("Figure 8 — Energy per Inference vs. Latency "
                 "(iso-energy contours in µJ shown in grey)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig8_energy_vs_latency.pdf",
                bbox_inches="tight", format="pdf")
    fig.savefig("/mnt/user-data/outputs/fig8_energy_vs_latency.png",
                bbox_inches="tight", dpi=200)
    print("  ✓  Figure 8 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating embedded benchmark figures...")
    fig4_latency_per_mcu()
    fig5_energy_per_mcu()
    fig6_pareto_acc_energy()
    fig7_latency_vs_params()
    fig8_energy_vs_latency()
    print("\nAll figures generated successfully.")
    print("Files written to /mnt/user-data/outputs/  (PDF + PNG each)")
