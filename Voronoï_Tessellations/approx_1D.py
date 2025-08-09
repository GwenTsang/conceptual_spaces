#!/usr/bin/env python3
"""
approx_models.py

A concise, robust script that:
 - Computes (or loads) a 2D membership grid for the two-segment geometry (deterministic quadrature or MC).
 - Collapses to 1D: mean_by_x and var_by_x (saved).
 - Fits Linear, Quadratic and PCHIP spline to mean_by_x.
 - Optionally fits a 4-parameter logistic for comparison.
 - Computes and prints metrics: MSE, RMSE, max_abs, R^2.
 - Saves plots and `results.npz`.

Usage examples:
  python approx_models.py --use_quadrature --quad_n 200
  python approx_models.py --n_mc 30000

This script is intentionally small and focused on the models you requested.
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, curve_fit
from scipy.special import expit


# ---------------------------
# Utilities
# ---------------------------

def compute_membership_quadrature(points, left_seg, right_seg, Nu=200, Nv=200, batch=500):
    left_seg = np.asarray(left_seg, dtype=np.float64)
    right_seg = np.asarray(right_seg, dtype=np.float64)
    xL = left_seg[0][0]
    yLmin = left_seg[0][1]
    yLmax = left_seg[1][1]
    xR = right_seg[0][0]
    yRmin = right_seg[0][1]
    yRmax = right_seg[1][1]

    u = np.linspace(yLmin, yLmax, Nu)
    v = np.linspace(yRmin, yRmax, Nv)
    left_samples = np.stack([np.full(Nu, xL), u], axis=1)
    right_samples = np.stack([np.full(Nv, xR), v], axis=1)

    M = points.shape[0]
    membership = np.empty(M, dtype=np.float64)
    denom = float(Nu * Nv)

    for i in range(0, M, batch):
        j = min(M, i + batch)
        pts = points[i:j]
        # (B,Nu) and (B,Nv)
        dL2 = np.sum((pts[:, None, :] - left_samples[None, :, :]) ** 2, axis=2)
        dR2 = np.sum((pts[:, None, :] - right_samples[None, :, :]) ** 2, axis=2)
        comp = (dL2[:, :, None] < dR2[:, None, :])
        counts = comp.reshape(pts.shape[0], -1).sum(axis=1)
        membership[i:j] = counts.astype(np.float64) / denom
    return membership


def compute_membership_mc(points, left_seg, right_seg, N_mc=20000, batch=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    left_seg = np.asarray(left_seg, dtype=np.float64)
    right_seg = np.asarray(right_seg, dtype=np.float64)
    tL = rng.random(N_mc)
    tR = rng.random(N_mc)
    samples_left = left_seg[0] + tL.reshape(-1, 1) * (left_seg[1] - left_seg[0])
    samples_right = right_seg[0] + tR.reshape(-1, 1) * (right_seg[1] - right_seg[0])

    M = points.shape[0]
    membership = np.empty(M, dtype=np.float64)

    for i in range(0, M, batch):
        j = min(M, i + batch)
        pts = points[i:j]
        dL2 = np.sum((pts[:, None, :] - samples_left[None, :, :]) ** 2, axis=2)
        dR2 = np.sum((pts[:, None, :] - samples_right[None, :, :]) ** 2, axis=2)
        fav = (dL2 < dR2).sum(axis=1)
        membership[i:j] = fav.astype(np.float64) / float(N_mc)
    return membership


# ---------------------------
# Models and metrics
# ---------------------------

def linear_fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coefs  # a, b


def quadratic_fit(x, y):
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coefs  # a, b, c


def sigmoid(x, k, x0, scale, shift):
    # use expit for numerical stability: expit(z) = 1/(1+exp(-z))
    z = k * (x - x0)
    return scale * expit(z) + shift


def safe_sigmoid(x, k, x0, scale, shift):
    return sigmoid(x, k, x0, scale, shift)


def metrics(y_true, y_pred):
    r = y_pred - y_true
    mse = float(np.mean(r**2))
    rmse = float(np.sqrt(mse))
    maxabs = float(np.max(np.abs(r)))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'mse': mse, 'rmse': rmse, 'max_abs': maxabs, 'r2': r2}


# ---------------------------
# Main
# ---------------------------

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # geometry
    left_seg = (np.array([args.left_x, args.seg_ymin]), np.array([args.left_x, args.seg_ymax]))
    right_seg = (np.array([args.right_x, args.seg_ymin]), np.array([args.right_x, args.seg_ymax]))

    xs = np.linspace(args.xmin, args.xmax, args.grid_x)
    ys = np.linspace(args.ymin, args.ymax, args.grid_y)
    Xg, Yg = np.meshgrid(xs, ys)
    points = np.vstack([Xg.ravel(), Yg.ravel()]).T

    if args.use_quadrature:
        print(f"Computing deterministic quadrature Nu=Nv={args.quad_n} ...")
        t0 = time.time()
        membership = compute_membership_quadrature(points, left_seg, right_seg, Nu=args.quad_n, Nv=args.quad_n, batch=args.batch)
        print("Quadrature time:", time.time() - t0)
        np.save(os.path.join(args.save_dir, 'membership_quadrature.npy'), membership)
    elif args.membership is not None and os.path.exists(args.membership):
        membership = np.load(args.membership).astype(np.float64)
        if membership.size != points.shape[0]:
            raise ValueError("membership size mismatch")
    else:
        Nmc = max(args.n_mc, 20000)
        print(f"Computing Monte-Carlo N_mc={Nmc} ...")
        t0 = time.time()
        membership = compute_membership_mc(points, left_seg, right_seg, N_mc=Nmc, batch=args.batch)
        print("MC time:", time.time() - t0)
        np.save(os.path.join(args.save_dir, 'membership_mc.npy'), membership)

    grid = membership.reshape(Xg.shape)
    mean_by_x = grid.mean(axis=0)
    var_by_x = grid.var(axis=0)
    np.save(os.path.join(args.save_dir, 'mean_by_x.npy'), mean_by_x)
    np.save(os.path.join(args.save_dir, 'var_by_x.npy'), var_by_x)

    # Linear
    lin_coefs = linear_fit(xs, mean_by_x)
    pred_lin = lin_coefs[0] * xs + lin_coefs[1]
    met_lin = metrics(mean_by_x, pred_lin)

    # Quadratic
    quad_coefs = quadratic_fit(xs, mean_by_x)
    pred_quad = quad_coefs[0] * xs**2 + quad_coefs[1] * xs + quad_coefs[2]
    met_quad = metrics(mean_by_x, pred_quad)

    # PCHIP
    pchip = PchipInterpolator(xs, mean_by_x)
    pred_pchip = pchip(xs)
    met_pchip = metrics(mean_by_x, pred_pchip)

    # Logistic (optional) - use least_squares and curve_fit for comparison
    # initial guess
    minv = float(mean_by_x.min())
    maxv = float(mean_by_x.max())
    scale0 = max(1e-6, maxv - minv)
    shift0 = minv
    midpoint = 0.5 * (minv + maxv)
    try:
        x0_guess = float(np.interp(midpoint, mean_by_x, xs))
    except Exception:
        x0_guess = 0.5 * (args.xmin + args.xmax)
    k0 = args.k_init
    p0 = [k0, x0_guess, scale0, shift0]
    bounds = ([1e-3, args.xmin, 1e-3, 0.0], [args.k_max, args.xmax, 1.0, 1.0])

    # weighting
    eps = 1e-12
    w = 1.0 / (var_by_x + eps)
    sqrtw = np.sqrt(w)

    def resid_ls(params):
        k, x0, scale, shift = params
        y = safe_sigmoid(xs, k, x0, scale, shift)
        return (y - mean_by_x) * sqrtw

    try:
        res = least_squares(resid_ls, p0, bounds=bounds, loss='soft_l1', max_nfev=20000)
        popt_ls = res.x
        pred_log_ls = safe_sigmoid(xs, *popt_ls)
        met_log_ls = metrics(mean_by_x, pred_log_ls)
    except Exception as e:
        popt_ls = None
        met_log_ls = None

    # curve_fit
    try:
        sigma = np.sqrt(var_by_x + eps)
        popt_cf, pcov_cf = curve_fit(lambda x, k, x0, scale, shift: safe_sigmoid(x, k, x0, scale, shift), xs, mean_by_x, p0=p0, bounds=bounds, sigma=sigma, maxfev=20000)
        pred_log_cf = safe_sigmoid(xs, *popt_cf)
        met_log_cf = metrics(mean_by_x, pred_log_cf)
    except Exception:
        popt_cf = None
        pcov_cf = None
        met_log_cf = None

    # Print results succinctly (matches the style you pasted)
    print("Linear fit coeffs:", np.array(lin_coefs))
    print("Quadratic fit coeffs:", np.array(quad_coefs))
    print("Linear metrics:", met_lin)
    print("Quadratic metrics:", met_quad)
    if popt_ls is not None:
        print("Logistic (LS) params:", popt_ls)
        print("Logistic (LS) metrics:", met_log_ls)
    if popt_cf is not None:
        print("Logistic (curve_fit) params:", popt_cf)
        print("Logistic (curve_fit) metrics:", met_log_cf)

    # Save results
    out = dict(
        lin_coefs=lin_coefs,
        quad_coefs=quad_coefs,
        met_lin=met_lin,
        met_quad=met_quad,
        met_pchip=met_pchip,
        pchip_pred=pred_pchip,
        logistic_ls_params=popt_ls,
        logistic_cf_params=popt_cf
    )
    np.savez(os.path.join(args.save_dir, 'results.npz'), **out)

    # Plots
    plt.figure(figsize=(8,4))
    plt.plot(xs, mean_by_x, label='mean_by_x', lw=2)
    plt.plot(xs, pred_lin, label='Linear')
    plt.plot(xs, pred_quad, label='Quadratic')
    plt.plot(xs, pred_pchip, '--', label='PCHIP')
    if popt_ls is not None:
        plt.plot(xs, pred_log_ls, ':', label='Logistic (LS)')
    if popt_cf is not None:
        plt.plot(xs, pred_log_cf, '-.', label='Logistic (CF)')
    plt.legend()
    plt.title('Model fits to mean_by_x')
    plt.xlabel('x')
    plt.ylabel('membership')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'fits.png'), dpi=150)
    plt.close()

    # Residuals histogram for best (lowest RMSE among linear/quadratic/pchip/logistic)
    candidates = {'linear': (pred_lin, met_lin), 'quadratic': (pred_quad, met_quad), 'pchip': (pred_pchip, met_pchip)}
    if popt_ls is not None:
        candidates['logistic_ls'] = (pred_log_ls, met_log_ls)
    if popt_cf is not None:
        candidates['logistic_cf'] = (pred_log_cf, met_log_cf)

    best = min(candidates.items(), key=lambda kv: kv[1][1]['rmse'])
    best_name = best[0]
    best_pred = best[1][0]
    best_metrics = best[1][1]
    print(f"Best model by RMSE: {best_name} -> RMSE={best_metrics['rmse']:.8f}")

    diff2d = np.tile(best_pred[np.newaxis,:], (args.grid_y, 1)) - grid
    mse2d = float(np.mean(diff2d**2))
    rmse2d = float(np.sqrt(mse2d))
    maxabs2d = float(np.max(np.abs(diff2d)))
    print(f"2D metrics (replicated): MSE={mse2d:.8f} RMSE={rmse2d:.8f} max_abs={maxabs2d:.8f}")

    # save residual histogram
    plt.figure(figsize=(6,4))
    plt.hist(diff2d.ravel(), bins=100)
    plt.title('Residuals histogram (replicated-best - target)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'residuals_hist.png'), dpi=150)
    plt.close()

    print('Outputs saved to', args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--membership', type=str, default=None)
    parser.add_argument('--use_quadrature', action='store_true')
    parser.add_argument('--quad_n', type=int, default=200)
    parser.add_argument('--n_mc', type=int, default=20000)
    parser.add_argument('--grid_x', type=int, default=200)
    parser.add_argument('--grid_y', type=int, default=80)
    parser.add_argument('--xmin', type=float, default=-3.0)
    parser.add_argument('--xmax', type=float, default=3.0)
    parser.add_argument('--ymin', type=float, default=-2.5)
    parser.add_argument('--ymax', type=float, default=2.5)
    parser.add_argument('--left_x', type=float, default=-1.0)
    parser.add_argument('--right_x', type=float, default=1.0)
    parser.add_argument('--seg_ymin', type=float, default=-10.0)
    parser.add_argument('--seg_ymax', type=float, default=10.0)
    parser.add_argument('--k_init', type=float, default=5.0)
    parser.add_argument('--k_max', type=float, default=200.0)
    parser.add_argument('--batch', type=int, default=500)
    parser.add_argument('--save_dir', type=str, default='approx_results')
    args = parser.parse_args()
    main(args)
