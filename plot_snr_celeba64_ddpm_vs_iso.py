import os, csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# 0) PATHS (your confirmed structure)
# -------------------------
DDPM_DIR = "Transient_Results/transient_results_celeba_64_0.0_t0_to_t150"  # DDPM
ISO_DIR  = "Transient_Results/transient_results_celeba_64_0.3_t0_to_t150"  # ISO diffusion

TIMESTEPS_REQ = [0,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]

OUTDIR = "outputs/snr/celeba64_ddpm_vs_iso"
os.makedirs(OUTDIR, exist_ok=True)

# -------------------------
# 1) Controls (start small, then full)
# -------------------------
MAX_IMAGES = None   # set to None for full 50000
USE_COMMON_IDS_ACROSS_METHODS = True  # recommended for fair comparison
WRITE_PER_IMAGE_CSV = False  # per-image CSV can be HUGE if MAX_IMAGES=None

EPS = 1e-8


# -------------------------
# 2) ECCV-ish plot style (clean)
# -------------------------
def set_eccv_style():
    plt.rcParams.update({
        "figure.figsize": (6.4, 3.9),
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.2,
        "lines.markersize": 5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# -------------------------
# 3) Helpers
# -------------------------
def tdir(root, t):
    return os.path.join(root, f"t_{t}")

def list_png_ids(folder):
    # filenames like "1234.png" -> id "1234.png" (keep full filename for exact match)
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    # numeric sort if possible
    def key_fn(x):
        base = os.path.splitext(x)[0]
        return int(base) if base.isdigit() else x
    files.sort(key=key_fn)
    return files

def load_img(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def snr_db(x0, xt):
    signal = float(np.mean(x0 ** 2))
    noise  = float(np.mean((xt - x0) ** 2))
    return 10.0 * np.log10((signal + EPS) / (noise + EPS))

class Welford:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        d2 = x - self.mean
        self.M2 += d * d2
    def finalize(self):
        if self.n <= 1:
            return self.mean, 0.0, 0.0
        var = self.M2 / (self.n - 1)
        std = float(np.sqrt(max(var, 0.0)))
        ci95 = 1.96 * std / np.sqrt(self.n)
        return float(self.mean), std, float(ci95)

def available_timesteps(root, requested):
    out = []
    for t in requested:
        if os.path.isdir(tdir(root, t)):
            out.append(t)
        else:
            print(f"[WARN] Missing folder: {tdir(root, t)} (skip)")
    return out


# -------------------------
# 4) Core computation
# -------------------------
def compute_method(method_name, root, timesteps, common_ids0=None):
    # ids from t_0
    ids0 = list_png_ids(tdir(root, 0))
    if common_ids0 is not None:
        ids = [x for x in ids0 if x in common_ids0]
    else:
        ids = ids0

    if MAX_IMAGES is not None:
        ids = ids[:MAX_IMAGES]

    # stats accumulators per timestep
    stats = {t: Welford() for t in timesteps}

    # optional per-image csv rows
    per_rows = []

    # stream over images: load x0 once per id, then loop timesteps
    for i, fname in enumerate(ids):
        p0 = os.path.join(tdir(root, 0), fname)
        if not os.path.isfile(p0):
            continue
        x0 = load_img(p0)

        for t in timesteps:
            pt = os.path.join(tdir(root, t), fname)
            if not os.path.isfile(pt):
                continue
            xt = load_img(pt)
            v = snr_db(x0, xt)
            stats[t].update(v)

            if WRITE_PER_IMAGE_CSV:
                per_rows.append({
                    "method": method_name,
                    "timestep": t,
                    "filename": fname,
                    "snr_db": v
                })

        if (i + 1) % 500 == 0:
            print(f"[INFO] {method_name}: processed {i+1}/{len(ids)}")

    # finalize to summary
    summary_rows = []
    ts_list, mean_list, ci_list = [], [], []

    for t in timesteps:
        mean, std, ci95 = stats[t].finalize()
        summary_rows.append({
            "method": method_name,
            "timestep": t,
            "n": stats[t].n,
            "snr_mean_db": mean,
            "snr_std_db": std,
            "snr_ci95_db": ci95
        })
        ts_list.append(t)
        mean_list.append(mean)
        ci_list.append(ci95)

    return (np.array(ts_list), np.array(mean_list), np.array(ci_list)), summary_rows, per_rows


# -------------------------
# 5) Plotting + CSV writing
# -------------------------
def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[SAVE] {path}")

def plot_two(curves, x_max, title, out_prefix):
    set_eccv_style()
    fig, ax = plt.subplots()

    for label, (ts, mean, ci) in curves.items():
        m = ts <= x_max
        ts2, mean2, ci2 = ts[m], mean[m], ci[m]
        ax.plot(ts2, mean2, marker="o", label=label)
        ax.fill_between(ts2, mean2 - ci2, mean2 + ci2, alpha=0.18)

    ax.set_xlabel("Diffusion timestep (t)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title(title)
    ax.set_xlim(0, x_max)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")

    out_png = os.path.join(OUTDIR, f"{out_prefix}_t0_to_t{x_max}.png")
    out_pdf = os.path.join(OUTDIR, f"{out_prefix}_t0_to_t{x_max}.pdf")
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {out_pdf}")

def main():
    # timesteps that actually exist (your folders do exist for these)
    ddpm_ts = available_timesteps(DDPM_DIR, TIMESTEPS_REQ)
    iso_ts  = available_timesteps(ISO_DIR,  TIMESTEPS_REQ)
    # use intersection of available timesteps
    timesteps = [t for t in ddpm_ts if t in iso_ts]
    print("[INFO] timesteps used:", timesteps)

    # common ids in t_0 across both methods (fairness)
    common_ids0 = None
    if USE_COMMON_IDS_ACROSS_METHODS:
        ddpm_ids0 = set(list_png_ids(tdir(DDPM_DIR, 0)))
        iso_ids0  = set(list_png_ids(tdir(ISO_DIR, 0)))
        common_ids0 = ddpm_ids0.intersection(iso_ids0)
        print(f"[INFO] common ids at t_0: {len(common_ids0)}")

    ddpm_curve, ddpm_summary, ddpm_per = compute_method("DDPM (r=0.0)", DDPM_DIR, timesteps, common_ids0)
    iso_curve,  iso_summary,  iso_per  = compute_method("ISO diffusion (r=0.3)", ISO_DIR, timesteps, common_ids0)

    # summary CSV (this is the main one you need)
    summary_csv = os.path.join(OUTDIR, "snr_summary_celeba64_ddpm_vs_iso.csv")
    write_csv(summary_csv, ddpm_summary + iso_summary, fieldnames=[
        "method","timestep","n","snr_mean_db","snr_std_db","snr_ci95_db"
    ])

    # optional per-image CSV
    if WRITE_PER_IMAGE_CSV:
        per_csv = os.path.join(OUTDIR, "snr_per_image_celeba64_ddpm_vs_iso.csv")
        write_csv(per_csv, ddpm_per + iso_per, fieldnames=[
            "method","timestep","filename","snr_db"
        ])

    curves = {
        "DDPM (r=0.0)": ddpm_curve,
        "ISO diffusion (r=0.3)": iso_curve,
    }

    plot_two(curves, 100,  "CelebA 64×64 — SNR over diffusion timesteps", "snr_celeba64_ddpm_vs_iso")
    plot_two(curves, 1000, "CelebA 64×64 — SNR over diffusion timesteps", "snr_celeba64_ddpm_vs_iso")

if __name__ == "__main__":
    main()
