import os
import numpy as np
import matplotlib.pyplot as plt

FIG_BG = "#faf7f2"
AX_BG = "#fffdf9"
PDF_COLOR = "#2a6f97"
CDF_COLOR = "#6a994e"
CI_COLOR = "#bc6c25"
PRIMARY_COLOR = "#355070"
SECONDARY_COLOR = "#8a5a44"


def _style_axes(ax):
    ax.set_facecolor(AX_BG)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#b8b2a7")


def _save_figure(outpath):
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, facecolor=FIG_BG)
    plt.close()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_psd_pdf(bin_centers, pdf, outpath, label="PSD"):
    fig, ax = plt.subplots(figsize=(6,4), facecolor=FIG_BG)
    ax.plot(bin_centers, pdf, lw=2.2, color=PDF_COLOR, label=label)
    _style_axes(ax)
    plt.xlabel("Pore radius (Å)")
    plt.ylabel("Probability density")
    plt.title("Pore Size Distribution (PDF)")
    plt.legend(frameon=False)
    _save_figure(outpath)

def plot_psd_cdf(bin_centers, cdf, outpath):
    fig, ax = plt.subplots(figsize=(6,4), facecolor=FIG_BG)
    ax.plot(bin_centers, cdf, lw=2.2, color=CDF_COLOR)
    _style_axes(ax)
    plt.xlabel("Pore radius (Å)")
    plt.ylabel("Cumulative fraction")
    plt.title("Pore Size Distribution (CDF)")
    plt.ylim(0, 1.0)
    _save_figure(outpath)

def plot_psd_with_ci(bin_centers, mean_pdf, lo_pdf, hi_pdf, outpath):
    fig, ax = plt.subplots(figsize=(6,4), facecolor=FIG_BG)
    ax.plot(bin_centers, mean_pdf, lw=2.2, color=PDF_COLOR, label="Mean PSD")
    ax.fill_between(bin_centers, lo_pdf, hi_pdf, alpha=0.22, color=CI_COLOR, label="95% CI")
    _style_axes(ax)
    plt.xlabel("Pore radius (Å)")
    plt.ylabel("Probability density")
    plt.title("PSD with Uncertainty")
    plt.legend(frameon=False)
    _save_figure(outpath)

def plot_sensitivity_curves(bin_centers, curves_dict, outpath, title):
    # curves_dict: {label: pdf_array}
    fig, ax = plt.subplots(figsize=(6,4), facecolor=FIG_BG)
    palette = [PDF_COLOR, CDF_COLOR, CI_COLOR, PRIMARY_COLOR, SECONDARY_COLOR]
    for index, (label, y) in enumerate(curves_dict.items()):
        ax.plot(bin_centers, y, lw=1.9, color=palette[index % len(palette)], label=label)
    _style_axes(ax)
    plt.xlabel("Pore radius (Å)")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.legend(fontsize=8, frameon=False)
    _save_figure(outpath)

def plot_timeseries(frame_idx, mean_r, porosity, outpath):
    fig, ax1 = plt.subplots(figsize=(7,4), facecolor=FIG_BG)
    ax1.plot(frame_idx, mean_r, color=PRIMARY_COLOR, lw=1.9)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Mean pore radius (Å)", color=PRIMARY_COLOR)
    ax1.tick_params(axis="y", colors=PRIMARY_COLOR)
    _style_axes(ax1)
    ax2 = ax1.twinx()
    ax2.plot(frame_idx, porosity, color=SECONDARY_COLOR, lw=1.6)
    ax2.set_ylabel("Porosity", color=SECONDARY_COLOR)
    ax2.tick_params(axis="y", colors=SECONDARY_COLOR)
    plt.title("Framewise Metrics")
    _save_figure(outpath)

def plot_dist_slice(dist_3d, outpath, axis="z", index=None, vmin=0, vmax=None):
    # dist_3d in Å
    if axis == "z":
        if index is None: index = dist_3d.shape[2] // 2
        img = dist_3d[:, :, index]
        ttl = f"EDT radius map (z-slice={index})"
    elif axis == "y":
        if index is None: index = dist_3d.shape[1] // 2
        img = dist_3d[:, index, :]
        ttl = f"EDT radius map (y-slice={index})"
    else:
        if index is None: index = dist_3d.shape[0] // 2
        img = dist_3d[index, :, :]
        ttl = f"EDT radius map (x-slice={index})"

    fig, ax = plt.subplots(figsize=(5,4), facecolor=FIG_BG)
    im = ax.imshow(img.T, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Local pore radius (Å)")
    _style_axes(ax)
    plt.title(ttl)
    _save_figure(outpath)