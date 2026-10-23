#!/usr/bin/env python3
"""R02/R03: full predicted-noise covariance during native CIFAR-10 reverse DDPM sampling.

Usage (put this script in REPO_ROOT/experiments/ or REPO_ROOT/scripts/):

    python experiments/transient_covarince_cifar10_ddpm.py \
        --ckpt checkpoints/ddpm/ckpt.pth --outdir outputs/cov_ddpm \
        --config configs/cifar10.yml --label DDPM --device cuda:0

    python experiments/transient_covarince_cifar10_ddpm.py \
        --ckpt checkpoints/endiff/ckpt.pth --outdir outputs/cov_endiff \
        --config configs/cifar10.yml --label EnDiff --device cuda:0 \
        --compare-outdir outputs/cov_ddpm

Uses the exact reverse UPDATE from the user-provided transient_r_theta_cifar10_ddpm.py:
  eps = model(x, t.float())
  x0 = clamp(sqrt(1/alpha_bar_t)*x - sqrt(1/alpha_bar_t-1)*eps, -1, 1)
  mean = [sqrt(alpha_bar_prev)*beta_step*x0 +
          sqrt(1-beta_step)*(1-alpha_bar_prev)*x] / (1-alpha_bar_t)
  x_prev = mean + 1[t>0]*sqrt(beta_step)*z

The covariance is across N *native reverse-state noise predictions*, at each
selected timestep, NOT covariance across pixels within one single image.

For CIFAR-10, each full covariance is 3072 x 3072. Every element is saved, without
sketches, random projections, or matrix downsampling (only HEATMAP rendering is
rescaled for display). Files are float64 NPY; sampled covariance uses N-1.
Products X^T X are computed in configurable torch float32 (default) or float64,
then accumulated in float64 CPU-backed NPY arrays. This is full-matrix covariance,
not an infinite-precision numerical calculation.

The original r_theta script labels model timestep 999 as '1000'. We retain that
*display label* for compatibility, but also save the true zero-based model index.

WARNING: Each float64 covariance file is 72 MiB for 3072 dimensions. The default
'auto' selection creates ~24 matrices (~1.7 GiB), plus plots/eigenvalues.
Analysis itself is checkpoint-only; it does NOT retrain either model.

This copy adds --save_images_dir: the final x_0 of every trajectory is also saved
as <i>.png (same format as main.py --sample --fid), so FID/PRDC and the covariance
come from the SAME reverse trajectories instead of two separate sampling runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imported lazily in main(), to allow --self-test to run without repo imports.
EPS = 1e-12


def dict2namespace(d: Dict) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, 'r', encoding='utf-8') as f:
        return dict2namespace(yaml.safe_load(f))


def resolve_config_path(user_config: str | None, ckpt_path: str) -> str:
    if user_config:
        path = Path(user_config)
        if not path.is_file():
            raise FileNotFoundError(f'--config not found: {path}')
        return str(path)
    candidate = Path(ckpt_path).resolve().parent / 'config.yml'
    if candidate.is_file():
        return str(candidate)
    for candidate in [Path('configs/cifar10_transient_50k.yml'), Path('configs/cifar10.yml')]:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError('Cannot locate CIFAR-10 config. Pass --config explicitly.')


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if s in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean: {v}')


def parse_timestamps(arg: str, T: int) -> List[int]:
    # '1000' is the user-provided script's label for true model index 999.
    if arg.strip().lower() == 'auto':
        raw = [T] + list(range(900, 199, -100)) + list(range(150, 9, -10)) + [0]
        raw = [x for x in raw if 0 <= x <= T]
    else:
        raw = [int(x.strip()) for x in arg.split(',') if x.strip()]
    labels = list(dict.fromkeys(raw))
    if not labels or any(x < 0 or x > T for x in labels):
        raise ValueError(f'--timestamps must be in [0,{T}]; top label {T} means index {T-1}.')
    actual = [T - 1 if x == T else x for x in labels]
    if len(set(actual)) != len(actual):
        raise ValueError(f'Duplicate model indices: do not specify both {T} and {T-1}.')
    return labels


def select_device(s: str) -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if s == 'auto' else torch.device(s)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    return device


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def load_model_from_checkpoint(config, ckpt_path: str, device, use_ema: bool):
    from models.diffusion import Model
    from models.ema import EMAHelper
    # Only load checkpoints you trust. Older repository checkpoints contain
    # Python state beyond a plain weights-only state dict.
    try:
        states = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        states = torch.load(ckpt_path, map_location=device)  # torch 1.x
    if not isinstance(states, (list, tuple)) or not states or not isinstance(states[0], dict):
        raise ValueError('Expected repository checkpoint [model_state, optimizer_state, epoch, step, (EMA)].')
    model = Model(config).to(device)
    model.load_state_dict(states[0], strict=True)
    if use_ema and getattr(config.model, 'ema', False):
        if len(states) <= 4:
            raise ValueError('EMA requested by --use_ema but checkpoint has no EMA weights.')
        ema = EMAHelper(mu=getattr(config.model, 'ema_rate', 0.9999))
        ema.register(model)
        ema.load_state_dict(states[-1])
        ema.ema(model)
    model.eval()
    return model


def build_betas(config, device):
    from runners.diffusion import get_beta_schedule
    betas_np = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    return torch.from_numpy(betas_np).float().to(device)


def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


class FullCovarianceAccumulator:
    """Accumulate uncentered second moments to disk; center only after all N samples.

    This correctly includes between-batch covariance (unlike naively averaging
    covariances computed on separate minibatches).
    """
    def __init__(self, path: Path, dim: int, matmul_dtype: torch.dtype):
        self.path, self.d, self.mm_dtype = path, dim, matmul_dtype
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.outer = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(dim, dim))
        self.outer.fill(0)
        self.outer.flush()

    def update(self, pred: torch.Tensor):
        x = pred.detach().reshape(pred.shape[0], -1)
        if x.shape[1] != self.d:
            raise ValueError(f'Expected vector dimension {self.d}, got {x.shape[1]}')
        if not torch.isfinite(x).all():
            raise FloatingPointError('Nonfinite predicted noise; check model/checkpoint.')
        self.sum += x.to(torch.float64).sum(0).cpu().numpy()
        # Full (d x d) product, not an approximation by projection.
        g = x.to(self.mm_dtype)
        product = (g.T @ g).to(torch.float64).cpu().numpy()
        self.outer[:] += product
        self.n += len(x)

    def finish(self) -> Path:
        if self.n < 2:
            raise ValueError('Need at least 2 predictions for covariance.')
        d, mu = self.d, self.sum / self.n
        # In-place centering in blocks to limit peak host RAM usage.
        for st in range(0, d, 128):
            en = min(st + 128, d)
            self.outer[st:en, :] -= self.sum[st:en, None] * mu[None, :]
            self.outer[st:en, :] /= (self.n - 1)
        self.outer.flush()
        del self.outer
        return self.path


def covariance_metrics(path: Path, sample_size: int, compute_eigen: bool, spectrum_dir: Path | None):
    C = np.load(path, mmap_mode='r')
    d = C.shape[0]
    diagonal = np.diag(C).copy()
    trace = float(diagonal.sum())
    diag_sum2 = float(np.dot(diagonal, diagonal))
    # Use the COMPLETE matrix: its Frobenius norm is sum of squared eigenvalues.
    fro2 = float(np.sum(np.square(C, dtype=np.float64)))
    off2 = max(0.0, fro2 - diag_sum2)
    centering_var = max(0., fro2 - trace * trace / d)
    if fro2 <= EPS:
        sphericity = float('nan')
        relative_anisotropy = float('nan')
    else:
        sphericity = trace * trace / (d * fro2)
        relative_anisotropy = math.sqrt(centering_var / fro2)
    di = np.maximum(diagonal, 0)
    sd = np.sqrt(di)
    denom = np.multiply.outer(sd, sd)
    corr = np.divide(C, denom, out=np.zeros((d, d), dtype=np.float64), where=denom > EPS)
    np.fill_diagonal(corr, 0)
    count_valid = int(np.sum(denom > EPS) - np.sum(di > EPS))
    corr_rms = float(np.sqrt(np.sum(corr * corr) / count_valid)) if count_valid > 0 else float('nan')
    identity_dist = float(math.sqrt(max(fro2 - 2 * trace + d, 0)))
    row = {
        'num_images': sample_size,
        'dim': d,
        'cov_trace': trace,
        'cov_mean_variance': trace / d,
        'cov_variance_std': float(diagonal.std()),
        'cov_variance_cv': float(diagonal.std() / diagonal.mean()) if diagonal.mean() > EPS else float('nan'),
        'cov_offdiag_rms': math.sqrt(off2 / (d * (d - 1))),
        'corr_offdiag_rms': corr_rms,
        'cov_frobenius_to_identity': identity_dist,
        'cov_relative_anisotropy_frobenius': relative_anisotropy,
        'spectral_sphericity': sphericity,
        'participation_ratio': trace * trace / fro2 if fro2 > EPS else float('nan'),
    }
    if compute_eigen:
        # Scipy's symmetric eigensolver avoids manual SVD of the N x d data.
        try:
            from scipy.linalg import eigvalsh
            eig = eigvalsh(np.asarray(C), check_finite=False)
        except ImportError:
            eig = np.linalg.eigvalsh(np.asarray(C))
        eig = np.asarray(eig, dtype=np.float64)
        # Small negative eigenvalues can arise from finite precision.
        small_neg = eig[eig < 0]
        row['negative_eigenvalue_count'] = int(len(small_neg))
        if len(small_neg) and eig[0] < -1e-5 * max(eig[-1], EPS):
            warnings.warn(f'{path.name}: materially negative eigenvalues; increase --cov-matmul-dtype to float64')
        eig = np.maximum(eig, 0)
        row.update({
            'lambda_min': float(eig[0]), 'lambda_median': float(np.median(eig)),
            'lambda_max': float(eig[-1]),
            'lambda_min_over_max': float(eig[0] / eig[-1]) if eig[-1] > EPS else float('nan'),
        })
        if spectrum_dir is not None:
            np.save(spectrum_dir / (path.stem.replace('covariance_', 'eigenvalues_') + '.npy'), eig)
    else:
        row.update({'negative_eigenvalue_count': '', 'lambda_min': '', 'lambda_median': '',
                    'lambda_max': '', 'lambda_min_over_max': ''})
    del C
    return row


def plot_single_matrix(path: Path, directory: Path, label: str, channels: int, size: int):
    C = np.load(path, mmap_mode='r')
    d = C.shape[0]
    cov = np.asarray(C)
    diag = np.diag(cov).copy()
    vmax = max(float(np.max(np.abs(C))), EPS)
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    # imshow downsamples only to DISPLAY; .npy retains all dxd entries.
    im = ax.imshow(cov, cmap='coolwarm', vmin=-vmax, vmax=vmax, interpolation='nearest', rasterized=True)
    fig.colorbar(im, ax=ax, label='Covariance')
    ax.set(xlabel='Flattened noise coordinate j', ylabel='Coordinate i',
           title=f'{label}: full covariance, {path.stem}')
    if channels == 3:
        for boundary in (size * size - .5, 2 * size * size - .5):
            ax.axhline(boundary, c='black', lw=.55)
            ax.axvline(boundary, c='black', lw=.55)
    fig.tight_layout()
    fig.savefig(directory / f'{path.stem}_heatmap.png', dpi=160)
    plt.close(fig)

    # For correlation, show off-diagonal structure without strong diagonal = 1.
    sd = np.sqrt(np.maximum(diag, 0))
    denom = np.multiply.outer(sd, sd)
    R = np.divide(cov, denom, out=np.zeros((d, d), dtype=np.float64), where=denom > EPS)
    np.fill_diagonal(R, 0)
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    im = ax.imshow(R, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest', rasterized=True)
    fig.colorbar(im, ax=ax, label='Correlation (diagonal hidden)')
    ax.set(xlabel='Flattened noise coordinate j', ylabel='Coordinate i',
           title=f'{label}: off-diagonal correlation, {path.stem}')
    if channels == 3:
        for boundary in (size * size - .5, 2 * size * size - .5):
            ax.axhline(boundary, c='black', lw=.55)
            ax.axvline(boundary, c='black', lw=.55)
    fig.tight_layout()
    fig.savefig(directory / f'{path.stem}_offdiag_correlation.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, channels, figsize=(4 * channels, 4), squeeze=False)
    vhi = max(float(np.max(diag)), EPS)
    for ch, ax in enumerate(axes.ravel()):
        im = ax.imshow(diag.reshape(channels, size, size)[ch], vmin=0, vmax=vhi, cmap='viridis')
        ax.set_title(f'Channel {ch} variance')
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.8, label='Predicted-noise coordinate variance')
    fig.suptitle(f'{label}: covariance diagonal reshaped as image, {path.stem}')
    fig.savefig(directory / f'{path.stem}_variance_maps.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_summary(rows: list[dict], out: Path, label: str, gaussian_metrics: dict | None):
    ordered = sorted(rows, key=lambda r: r['model_timestep'], reverse=True)
    keys = [
        ('cov_trace', 'Covariance trace'),
        ('cov_mean_variance', 'Mean diagonal variance'),
        ('cov_variance_cv', 'Diagonal variance CV'),
        ('cov_offdiag_rms', 'Off-diagonal covariance RMS'),
        ('corr_offdiag_rms', 'Off-diagonal correlation RMS'),
        ('cov_relative_anisotropy_frobenius', 'Relative distance from scaled identity'),
        ('spectral_sphericity', 'Spectral sphericity (1 = isotropic)'),
        ('cov_frobenius_to_identity', 'Frobenius distance from identity'),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 15))
    for ax, (key, title) in zip(axes.flat, keys):
        xs = [r['model_timestep'] for r in ordered]
        ys = [r[key] for r in ordered]
        ax.plot(xs, ys, marker='o', ms=3, lw=1.4, label=label)
        if gaussian_metrics is not None and key in gaussian_metrics:
            ax.axhline(float(gaussian_metrics[key]), linestyle=':', color='gray', label='Finite-N Gaussian reference')
        ax.invert_xaxis()
        ax.set(xlabel='Actual reverse timestep (sampling moves left to right)', title=title)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'R02_R03_all_metrics.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5))
    srows = [r for r in ordered if r['lambda_max'] != '']
    if srows:
        ax.plot([r['model_timestep'] for r in srows], [r['lambda_min'] for r in srows], label='Smallest eigenvalue')
        ax.plot([r['model_timestep'] for r in srows], [r['lambda_median'] for r in srows], label='Median eigenvalue')
        ax.plot([r['model_timestep'] for r in srows], [r['lambda_max'] for r in srows], label='Largest eigenvalue')
        ax.invert_xaxis(); ax.grid(alpha=.2)
        ax.set(xlabel='Actual reverse timestep', ylabel='Eigenvalue', title=f'{label}: covariance eigenvalues')
        ax.legend(); fig.tight_layout(); fig.savefig(out / 'R03_extreme_eigenvalues.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5))
    for r in srows:
        eig_path = out / 'eigenvalues' / f'eigenvalues_t{r["timestep_label"]}.npy'
        if eig_path.exists():
            eig = np.load(eig_path)[::-1]
            ax.plot(np.arange(1, len(eig) + 1), eig, lw=.9, alpha=.7, label=f't={r["timestep_label"]}')
    if srows:
        ax.set(xlabel='Eigenvalue rank (largest first)', ylabel='Eigenvalue', title=f'{label}: full eigenvalue spectra')
        ax.grid(alpha=.2)
        if len(srows) <= 12: ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(out / 'R03_eigenvalue_spectra.png', dpi=180)
    plt.close(fig)


def compare_runs(out: Path, other_dir: Path, label: str, plot_limit: int):
    mine = json.loads((out / 'analysis_settings.json').read_text(encoding='utf-8'))
    theirs = json.loads((other_dir / 'analysis_settings.json').read_text(encoding='utf-8'))
    required = ('seed','batch_size','num_images_per_t','num_timesteps','image_dimensions','sampler','use_ema','timestamps')
    mismatch = [k for k in required if mine.get(k) != theirs.get(k)]
    if mismatch:
        raise ValueError(f'Cannot compare unpaired runs: metadata fields differ: {mismatch}')
    with (out / 'covariance_summary.csv').open(newline='') as f:
        a = list(csv.DictReader(f))
    with (other_dir / 'covariance_summary.csv').open(newline='') as f:
        b = list(csv.DictReader(f))
    am = {int(r['timestep_label']):r for r in a}
    bm = {int(r['timestep_label']):r for r in b}
    common = sorted(set(am) & set(bm), reverse=True)
    target = out / 'comparison_with_other_run'
    target.mkdir(exist_ok=True)
    metrics = ['cov_mean_variance','cov_variance_cv','corr_offdiag_rms',
               'cov_relative_anisotropy_frobenius','spectral_sphericity','cov_frobenius_to_identity']
    fig,axes=plt.subplots(3,2,figsize=(12,12))
    for ax,key in zip(axes.flat,metrics):
        for rr,lbl in [(b,theirs.get('label','Other model')),(a,label)]:
            ax.plot([float(r['model_timestep']) for r in rr],
                    [float(r[key]) for r in rr], marker='o',ms=3,label=lbl)
        ax.invert_xaxis(); ax.grid(alpha=.2); ax.set(xlabel='Actual reverse timestep',title=key)
    axes[0,0].legend(); fig.tight_layout()
    fig.savefig(target / 'DDPM_EnDiff_R02_R03_comparison.png',dpi=180)
    plt.close(fig)

    # Save exact full matrix differences, and plot same-color-scale heatmaps.
    # Limit *plotted* snapshots to keep plots readable; all source matrices saved.
    if plot_limit:
        selected = list(np.unique(np.rint(np.linspace(0,len(common)-1,min(plot_limit,len(common)))).astype(int)))
    else:
        selected = []
    for ix in selected:
        t=common[ix]
        ca=np.load(out/'covariances'/f'covariance_t{t}.npy',mmap_mode='r')
        cb=np.load(other_dir/'covariances'/f'covariance_t{t}.npy',mmap_mode='r')
        if ca.shape!=cb.shape: raise ValueError(f'Covariance shape mismatch at t={t}.')
        diff=np.asarray(ca)-np.asarray(cb)
        np.save(target/f'covariance_delta_{label}_minus_other_t{t}.npy', diff)
        vmax=max(float(np.max(np.abs(ca))),float(np.max(np.abs(cb))),EPS)
        delta_max=max(float(np.max(np.abs(diff))),EPS)
        fig,axs=plt.subplots(1,3,figsize=(15,4.7))
        for ax,c,ttl,scale in [(axs[0],cb,theirs.get('label','Other'),vmax),
                               (axs[1],ca,label,vmax),(axs[2],diff,f'{label} minus other',delta_max)]:
            im=ax.imshow(c,cmap='coolwarm',vmin=-scale,vmax=scale,interpolation='nearest',rasterized=True)
            ax.set(title=f'{ttl}, t={t}',xlabel='Coordinate',ylabel='Coordinate')
            fig.colorbar(im,ax=ax,shrink=.65)
        fig.tight_layout();fig.savefig(target/f'paired_heatmaps_t{t}.png',dpi=140);plt.close(fig)
    # Compare COMPLETE eigenvalue spectra (raw and trace-normalized),
    # avoiding a misleading apparent isotropy improvement from scale alone.
    if selected:
        fig, axes = plt.subplots(len(selected), 2, figsize=(13, 3.7 * len(selected)), squeeze=False)
        for row_i, ix in enumerate(selected):
            t = common[ix]
            spectra = []
            for folder, name in [(other_dir, theirs.get('label','Other')),(out,label)]:
                eig_path = folder / 'eigenvalues' / f'eigenvalues_t{t}.npy'
                if eig_path.is_file():
                    eig = np.load(eig_path)[::-1]
                    spectra.append((eig, name))
            for eig, name in spectra:
                axes[row_i, 0].plot(np.arange(1, len(eig)+1), eig, lw=1, label=name)
                axes[row_i, 1].plot(np.arange(1, len(eig)+1),
                                     eig / max(float(eig.mean()), EPS), lw=1, label=name)
            axes[row_i, 0].set_title(f'Raw covariance eigenvalues, t={t}')
            axes[row_i, 1].set_title(f'Eigenvalues / mean eigenvalue, t={t}')
            for ax in axes[row_i]:
                ax.set_xlabel('Eigenvalue rank (largest first)')
                ax.grid(alpha=.2)
                if spectra:ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(target / 'DDPM_EnDiff_R03_eigenvalue_spectrum_comparison.png', dpi=165)
        plt.close(fig)
    comparison=[]
    for t in common:
        r={'timestep_label':t,'model_timestep':int(am[t]['model_timestep'])}
        for key in metrics:
            r[f'delta_{key}_{label}_minus_other']=float(am[t][key])-float(bm[t][key])
        comparison.append(r)
    save_csv(target/'R02_R03_metric_deltas.csv',comparison)
    print(f'[DONE] Compared to {other_dir}. Figures: {target}')


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'covariance_t2.npy'
        acc=FullCovarianceAccumulator(p,4,torch.float64)
        gen=np.random.default_rng(5)
        x=gen.normal(size=(49,4))
        for z in np.array_split(x,7):acc.update(torch.from_numpy(z))
        acc.finish()
        got=np.load(p)
        ref=np.cov(x,rowvar=False,ddof=1)
        assert np.allclose(got,ref,atol=1e-12),np.max(np.abs(got-ref))
        ms=covariance_metrics(p,49,True,None)
        assert np.isfinite(ms['spectral_sphericity'])
        assert abs(ms['cov_trace']-np.trace(ref))<1e-10
        assert parse_timestamps('1000,900,100,0',1000)==[1000,900,100,0]
        try:parse_timestamps('1000,999',1000);raise AssertionError('Duplicate alias not rejected')
        except ValueError:pass
    print('[SELF-TEST PASS] Chunked full covariance equals numpy.cov; spectrum and timestep parsing valid.')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt',required=False,help='Original repository checkpoint (.pth)')
    p.add_argument('--outdir',required=False,help='New output directory for this model')
    p.add_argument('--config',default=None)
    p.add_argument('--device',default='auto')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--batch_size',type=int,default=256)
    p.add_argument('--num_images_per_t',type=int,default=10000)
    p.add_argument('--timestamps',default='auto',help='auto or comma-separated: 1000,900,...,0')
    p.add_argument('--use_ema',type=str2bool,default=True)
    p.add_argument('--label',default='Model',help='Display label, e.g. DDPM or EnDiff')
    p.add_argument('--cov-matmul-dtype',choices=('float32','float64'),default='float32',
                   help='float64 for more precise products; substantially slower on GPUs with weak FP64')
    p.add_argument('--skip-eigenvalues',action='store_true',help='Skip expensive full eigen decompositions')
    p.add_argument('--heatmap-timestamps',default='auto',help='auto or comma-separated saved labels, e.g. 1000,500,100,0')
    p.add_argument('--compare-outdir',type=Path,default=None,help='Earlier model output to compare to (e.g. DDPM)')
    p.add_argument('--compare-heatmaps',type=int,default=4,help='Number of pairwise matrix heatmaps')
    p.add_argument('--overwrite',action='store_true',help='Allow overwriting files in a nonempty output folder')
    p.add_argument('--no-gaussian-reference',action='store_true',help='Skip finite-sample Gaussian null covariance')
    p.add_argument('--self-test',action='store_true',help='Check formulas without loading repository modules')
    p.add_argument('--save_images_dir',type=Path,default=None,
                   help='Also save the final sample of every trajectory here as <i>.png (for FID/PRDC)')
    a=p.parse_args()
    if a.self_test:return self_test()
    if not a.ckpt or not a.outdir:p.error('--ckpt and --outdir are required unless --self-test')
    if a.batch_size<1 or a.num_images_per_t<2 or a.compare_heatmaps<0:p.error('Invalid counts')
    if not Path(a.ckpt).is_file():p.error(f'Checkpoint not found: {a.ckpt}')
    out=Path(a.outdir)
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        p.error(f'Output directory is not empty: {out}. Choose another path or use --overwrite.')
    out.mkdir(parents=True,exist_ok=True)
    covdir=out/'covariances'; eigdir=out/'eigenvalues';figdir=out/'matrix_figures'
    if a.save_images_dir is not None:
        old_pngs=sorted(a.save_images_dir.glob('*.png')) if a.save_images_dir.is_dir() else []
        if old_pngs and not a.overwrite:
            p.error(f'Image directory is not empty: {a.save_images_dir}. Use --overwrite to replace its PNGs.')
        for f in old_pngs: f.unlink()
        a.save_images_dir.mkdir(parents=True,exist_ok=True)
    for q in [covdir,eigdir,figdir]:q.mkdir(exist_ok=True)
    cfgpath=resolve_config_path(a.config,a.ckpt)
    cfg=load_config(cfgpath)
    device=select_device(a.device)
    # Model initialization consumes torch RNG state; reseed immediately before sampling
    # so separate DDPM and EnDiff runs with the same flags have common random draws.
    model=load_model_from_checkpoint(cfg,a.ckpt,device,a.use_ema)
    betas=build_betas(cfg,device)
    T=len(betas); labels=parse_timestamps(a.timestamps,T)
    actual_to_label={T-1 if label==T else label:label for label in labels}
    channels=int(cfg.data.channels);size=int(cfg.data.image_size);d=channels*size*size
    if channels!=3 or size!=32:
        warnings.warn(f'Expected CIFAR-10 3x32x32, got {channels}x{size}x{size}; proceeding with d={d}.')
    if a.num_images_per_t<=d:
        warnings.warn(f'N={a.num_images_per_t} <= d={d}: sample covariance cannot have full rank.')
    if a.heatmap_timestamps.lower()=='auto':
        requested=[T,round(.5*T),round(.1*T),0]
        heatmap_labels=[x for x in requested if x in labels]
        if not heatmap_labels:heatmap_labels=[labels[0],labels[-1]]
    else:
        heatmap_labels=[int(x.strip()) for x in a.heatmap_timestamps.split(',') if x.strip()]
        if set(heatmap_labels)-set(labels):p.error('Heatmap labels must be included in --timestamps.')
    dtype=torch.float64 if a.cov_matmul_dtype=='float64' else torch.float32
    if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False
    acc={label:FullCovarianceAccumulator(covdir/f'covariance_t{label}.npy',d,dtype) for label in labels}
    settings={'script':'transient_covarince_cifar10_ddpm.py','ckpt':str(Path(a.ckpt).resolve()),
              'config':str(Path(cfgpath).resolve()),'label':a.label,'num_images_per_t':a.num_images_per_t,
              'batch_size':a.batch_size,'seed':a.seed,'use_ema':a.use_ema,
              'sampler':'original_transient_r_theta_cifar10_ddpm_ancestral',
              'num_timesteps':T,'image_dimensions':[channels,size,size],
              'timestamps':labels,'model_timestep_for_label':{str(v):int(k) for k,v in actual_to_label.items()},
              'covariance':'sample centered, denominator N-1, full dimension; no projection',
              'matmul_dtype':a.cov_matmul_dtype,
              'covariance_storage_dtype':'float64',
              'centered_population':'across generated images at each native reverse state',
              'selected_heatmaps':heatmap_labels,
              'saved_images_dir':str(a.save_images_dir.resolve()) if a.save_images_dir else None}
    (out/'analysis_settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print(f'[INFO] model={a.ckpt} | device={device} | N={a.num_images_per_t} | d={d}')
    print(f'[INFO] snapshot labels={labels} | matrix storage: approx {len(labels)*8*d*d/2**30:.2f} GiB')
    set_seed(a.seed,device)
    seq=list(range(T));seq_next=[-1]+seq[:-1]
    remaining=a.num_images_per_t
    if a.save_images_dir is not None:
        import torchvision.utils as tvu
        from datasets import inverse_data_transform
    img_id=0
    with torch.no_grad():
        for _ in tqdm(range(math.ceil(a.num_images_per_t/a.batch_size)),desc='R02/R03 native reverse sampling'):
            n=min(a.batch_size,remaining)
            x=torch.randn(n,channels,size,size,device=device)
            for i,j in zip(reversed(seq),reversed(seq_next)):
                t=torch.full((n,),i,device=device,dtype=torch.long)
                next_t=torch.full((n,),j,device=device,dtype=torch.long)
                at=compute_alpha(betas,t);atm1=compute_alpha(betas,next_t)
                beta_t=1-at/atm1
                eps_theta=model(x,t.float())
                if i in actual_to_label:
                    acc[actual_to_label[i]].update(eps_theta)
                x0_from_e=(1.0/at).sqrt()*x-(1.0/at-1.0).sqrt()*eps_theta
                x0_from_e=torch.clamp(x0_from_e,-1.0,1.0)
                mean_eps=(atm1.sqrt()*beta_t*x0_from_e +
                          ((1.0-beta_t).sqrt()*(1.0-atm1))*x)/(1.0-at)
                noise=torch.randn_like(x)  # also draw at t=0, matching provided script
                mask=(t!=0).float().view(-1,1,1,1)
                x=mean_eps+mask*torch.exp(.5*beta_t.log())*noise
            if a.save_images_dir is not None:
                # Final sample of these trajectories, saved like Diffusion.sample_fid.
                imgs=inverse_data_transform(cfg,x)
                for k in range(n):
                    tvu.save_image(imgs[k],str(a.save_images_dir/f'{img_id}.png'))
                    img_id+=1
            remaining-=n
    print('[INFO] All predictions collected; finalizing centered sample covariance matrices...')
    for label in labels:
        if acc[label].n!=a.num_images_per_t:
            raise RuntimeError(f't={label}: collected {acc[label].n} instead of {a.num_images_per_t}.')
        acc[label].finish()
    del acc
    gaussian_metrics=None
    if not a.no_gaussian_reference:
        print('[INFO] Computing one finite-sample N(0,I) covariance reference...')
        # Run gaussian reference with same N and d, independent CPU generator.
        gen=torch.Generator(device='cpu').manual_seed(a.seed+9001)
        gacc=FullCovarianceAccumulator(out/'gaussian_reference_covariance.npy',d,dtype)
        for st in range(0,a.num_images_per_t,a.batch_size):
            n=min(a.batch_size,a.num_images_per_t-st)
            gacc.update(torch.randn((n,d),generator=gen).to(device))
        gacc.finish()
        gaussian_metrics=covariance_metrics(out/'gaussian_reference_covariance.npy',a.num_images_per_t,
                                            compute_eigen=not a.skip_eigenvalues,spectrum_dir=None)
        (out/'gaussian_reference_metrics.json').write_text(json.dumps(gaussian_metrics,indent=2))
    rows=[]
    for label in labels:
        path=covdir/f'covariance_t{label}.npy'
        ms=covariance_metrics(path,a.num_images_per_t,not a.skip_eigenvalues,eigdir)
        r={'timestep_label':label,'model_timestep':T-1 if label==T else label,**ms}
        rows.append(r)
        print(f'[OK] t={label}: trace/d={r["cov_mean_variance"]:.5f} '
              f'sphericity={r["spectral_sphericity"]:.5f} corr_RMS={r["corr_offdiag_rms"]:.5f}',flush=True)
        if label in heatmap_labels:
            plot_single_matrix(path,figdir,a.label,channels,size)
    save_csv(out/'covariance_summary.csv',rows)
    plot_summary(rows,out,a.label,gaussian_metrics)
    if a.compare_outdir is not None: compare_runs(out,a.compare_outdir,a.label,a.compare_heatmaps)
    print(f'[DONE] R02/R03 results in {out.resolve()}')

if __name__=='__main__':
    main()
