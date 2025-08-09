"""
fit_voronoi_membership.py

Purpose:
  - Recompute (or load) the Monte-Carlo Voronoi membership field for target region r1 (uses cupy GPU sampling).
  - Define a differentiable parametric model that approximates that field.
  - Optimize model parameters (PyTorch on CUDA) to minimize MSE between model output and target.
  - Save results and plots.

Run:
  python fit_voronoi_membership.py

Requirements:
  - Python 3.8+
  - numpy, matplotlib, scipy, shapely, scipy.spatial
  - cupy (for Monte Carlo target generation)
  - torch (CUDA build recommended)
  - tqdm
"""

import os
import time
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import itertools
from matplotlib.patches import Polygon as MPLPolygon
import random

# GPU Monte Carlo code uses cupy; optimization uses torch.
try:
    import cupy as cp
except Exception as e:
    raise RuntimeError("This script uses cupy for the Monte Carlo target generation. "
                       "Install cupy-cudaXX matching your CUDA version. Error: " + str(e))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm

# --------------------------
# Problem geometry (copied from your code)
# --------------------------
vertices = {
    'r1': [(0.50000000, 0.25000000), (1.50000000, 0.25000000), (1.00000000, 1.00000000)],
    'r2': [(3.50000000, 0.25000000), (4.50000000, 0.25000000), (4.00000000, 1.00000000)],
    'r3': [(0.50000000, 3.25000000), (1.50000000, 3.25000000), (1.00000000, 4.00000000)],
    'r4': [(3.50000000, 3.25000000), (4.50000000, 3.25000000), (4.00000000, 4.00000000)],
}
prototypical_regions = {name: Polygon(verts) for name, verts in vertices.items()}
target_index = 0  # r1

# --------------------------
# Grid / plotting parameters (same domain as your code)
# --------------------------
xlim = (-0.5, 5.5)
ylim = (-0.5, 5.0)
width = xlim[1] - xlim[0]
height = ylim[1] - ylim[0]
grid_size_x = 240
grid_size_y = int(grid_size_x * (height / width))  # same aspect scaling
xs = np.linspace(xlim[0], xlim[1], grid_size_x)
ys = np.linspace(ylim[0], ylim[1], grid_size_y)
Xg, Yg = np.meshgrid(xs, ys)
points_to_test = np.vstack([Xg.ravel(), Yg.ravel()]).T
num_points = points_to_test.shape[0]
print(f"Grid: {grid_size_x} x {grid_size_y} => {num_points} points")

# --------------------------
# Cupy-based Monte Carlo membership (re-using your logic)
# --------------------------
CLIP_BBOX = Polygon([(-1.0, -1.0), (11.0, -1.0), (11.0, 11.0), (-1.0, 11.0)])

vertices_gpu = {name: cp.array(verts, dtype=cp.float32) for name, verts in vertices.items()}

def sample_point_in_triangle_gpu(triangle_vertices, n_samples):
    """Uniform sampling in triangle via barycentric coordinates (cupy)."""
    v1, v2, v3 = triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]
    r = cp.random.random((n_samples, 2), dtype=cp.float32)
    # make uniform by reflecting over diagonal for points with sum>1
    s = r[:, 0] + r[:, 1]
    cond = s > 1.0
    r1 = cp.where(cond, 1.0 - r[:, 0], r[:, 0]).astype(cp.float32)
    r2 = cp.where(cond, 1.0 - r[:, 1], r[:, 1]).astype(cp.float32)
    r1 = r1.reshape(-1, 1)
    r2 = r2.reshape(-1, 1)
    r3 = 1.0 - r1 - r2
    points = r1 * v1 + r2 * v2 + r3 * v3
    return points  # shape (n_samples, 2)

def compute_membership_monte_carlo_gpu(points_grid, n_samples=2000, batch_points=4096):
    """Compute membership for all grid points using GPU Monte Carlo with cupy.

    points_grid: numpy array (N,2)
    n_samples: number of MC samples per region (per draw) used to decide which region is closest.
               Note: this follows your original approach: sample one point per region per draw.
    batch_points: how many grid points to evaluate in one chunk to limit memory.
    Returns: numpy array of shape (N,) of membership probabilities for target_index region.
    """
    # We will emulate your algorithm: for each grid point we run n_samples draws,
    # each draw samples one point inside each triangle -> get 4 distances -> argmin -> is it target?
    N = points_grid.shape[0]
    out = np.zeros(N, dtype=np.float32)
    points_gpu = cp.asarray(points_grid.astype(np.float32))

    # Pre-sample per-region point clouds of size n_samples: for each draw we will use the corresponding row
    config_samples = cp.empty((n_samples, 4, 2), dtype=cp.float32)
    for i in range(4):
        triangle_verts = vertices_gpu[f'r{i+1}']
        config_samples[:, i, :] = sample_point_in_triangle_gpu(triangle_verts, n_samples)
    # config_samples shape: (n_samples, 4, 2)

    # evaluate in batches of grid points
    for start in range(0, N, batch_points):
        end = min(N, start + batch_points)
        batch = points_gpu[start:end]  # shape (B,2)
        B = end - start
        # expand shapes: (B, 1, 1, 2) - grid points; (1, n_samples, 4, 2)
        dists = cp.linalg.norm(batch.reshape(B, 1, 1, 2) - config_samples.reshape(1, n_samples, 4, 2), axis=3)
        # dists shape: (B, n_samples, 4)
        closest = cp.argmin(dists, axis=2)  # (B, n_samples)
        favorable_counts = cp.sum(closest == target_index, axis=1)  # (B,)
        membership = (favorable_counts / n_samples).astype(cp.float32)
        out[start:end] = cp.asnumpy(membership)
    return out

# --------------------------
# Differentiable model in PyTorch
# --------------------------
class SoftMinSoftmaxModel(nn.Module):
    """
    Model summary:
     - anchors_per_region: fixed anchor points S_{i,k} sampled inside each triangle (not trained)
     - param log_beta_anchor: controls softmin across anchors for each region (positive)
     - param log_gamma: controls softmax sharpness across regions (positive)
     - optional per-region linear scale/offset to allow small rescaling of distances
     - final monotonic transform: sigma * sigmoid(scale*(u - shift)) optionally (we will keep simple)
    """

    def __init__(self, anchors, device='cuda', use_region_affine=False, use_sigmoid_transform=False):
        """
        anchors: list of 4 numpy arrays of shape (K,2) (anchors per region)
        """
        super().__init__()
        self.device = device
        # Stack anchors into tensors for fast distance computation
        # anchors_tensor shape: (4, K, 2)
        anchors_stack = np.stack(anchors, axis=0).astype(np.float32)
        self.anchors = torch.tensor(anchors_stack, device=device)  # not a parameter
        K = anchors_stack.shape[1]

        # trainable parameters
        # ensure positivity via exponent of unconstrained params
        self.log_beta_anchor = nn.Parameter(torch.tensor(0.0, device=device))  # start at beta_anchor=1.0
        self.log_gamma = nn.Parameter(torch.tensor(1.0, device=device))       # start gamma=e^1~2.7

        self.use_region_affine = use_region_affine
        if use_region_affine:
            # per-region scale and offset (small init)
            self.region_scale = nn.Parameter(torch.ones(4, device=device) * 1.0)
            self.region_offset = nn.Parameter(torch.zeros(4, device=device))

        self.use_sigmoid_transform = use_sigmoid_transform
        if use_sigmoid_transform:
            # final monotonic transform parameters
            self.out_scale = nn.Parameter(torch.tensor(1.0, device=device))   # multiply probability
            self.out_shift = nn.Parameter(torch.tensor(0.0, device=device))
            self.out_sig_a = nn.Parameter(torch.tensor(10.0, device=device))  # sigmoid sharpness

    def forward(self, x):  # x shape: (B,2)
        """
        Returns p_target for each point in x; values in [0,1]
        """
        # x: (B,2)
        B = x.shape[0]
        anchors = self.anchors  # (4, K, 2)
        # compute squared distances: expand dims
        # x_exp: (B, 1, 1, 2), anchors: (1, 4, K, 2)
        x_exp = x.view(B, 1, 1, 2)
        anchors_exp = anchors.unsqueeze(0)  # (1,4,K,2)
        diffs = x_exp - anchors_exp  # (B,4,K,2)
        d2 = torch.sum(diffs * diffs, dim=3)  # (B,4,K) squared distances

        # softmin over anchors: use log-sum-exp with parameter beta_anchor
        beta_anchor = torch.exp(self.log_beta_anchor) + 1e-8  # ensure positive
        # Soft-min approximation to min_k d2_{ik}:
        # softmin(d2) = - (1/beta) * logsumexp(-beta * d2_k)
        neg_beta = -beta_anchor
        # compute logsumexp along K axis
        # shape (B,4)
        lse = torch.logsumexp(neg_beta * d2, dim=2)
        softmin = - (1.0 / beta_anchor) * lse  # shape (B,4)
        # optional per-region affine
        if self.use_region_affine:
            softmin = softmin * self.region_scale.unsqueeze(0) + self.region_offset.unsqueeze(0)

        # now convert distances to "scores": lower distance -> higher score
        gamma = torch.exp(self.log_gamma) + 1e-8
        scores = - gamma * softmin  # shape (B,4)
        # softmax across regions -> probabilities
        probs = torch.softmax(scores, dim=1)  # (B,4)
        p_target = probs[:, 0]  # target is region index 0 (r1)

        if self.use_sigmoid_transform:
            # allow a final small monotonic transform:
            a = torch.abs(self.out_sig_a) + 1e-6
            shifted = a * (p_target - self.out_shift)
            p_target = self.out_scale * torch.sigmoid(shifted)
            # clamp to [0,1]
            p_target = torch.clamp(p_target, 0.0, 1.0)

        return p_target

# --------------------------
# Helpers: sample anchors inside each triangle (numpy)
# --------------------------
def sample_points_in_triangle_numpy(verts, n):
    """Uniform sample `n` points inside triangle with vertices verts (3x2 numpy)."""
    v1 = np.array(verts[0], dtype=np.float32)
    v2 = np.array(verts[1], dtype=np.float32)
    v3 = np.array(verts[2], dtype=np.float32)
    r = np.random.rand(n, 2).astype(np.float32)
    s = r[:, 0] + r[:, 1]
    cond = s > 1.0
    r1 = np.where(cond, 1.0 - r[:, 0], r[:, 0]).astype(np.float32)
    r2 = np.where(cond, 1.0 - r[:, 1], r[:, 1]).astype(np.float32)
    r1 = r1.reshape(-1, 1)
    r2 = r2.reshape(-1, 1)
    r3 = 1.0 - r1 - r2
    pts = r1 * v1 + r2 * v2 + r3 * v3
    return pts  # shape (n,2)

# --------------------------
# Training / execution
# --------------------------
def fit_model(
    membership_path="membership.npy",
    recompute_target=False,
    n_mc_samples=2000,
    anchors_per_region=400,
    batches_for_training=64,
    n_epochs=800,
    lr=1e-2,
    device=None,
    use_region_affine=False,
    use_sigmoid_transform=False,
    save_dir="fit_results"
):
    os.makedirs(save_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Compute or load Monte Carlo target
    if recompute_target or not os.path.exists(membership_path):
        print("Computing Monte Carlo target membership (this can take a while)...")
        t0 = time.time()
        membership_flat = compute_membership_monte_carlo_gpu(points_to_test, n_samples=n_mc_samples, batch_points=4096)
        np.save(membership_path, membership_flat)
        t1 = time.time()
        print(f"Done Monte Carlo in {t1 - t0:.1f}s; saved to {membership_path}")
    else:
        print("Loading existing membership from", membership_path)
        membership_flat = np.load(membership_path).astype(np.float32)

    # Reshape to grid for plotting later
    membership_grid = membership_flat.reshape(Xg.shape)

    # 2) sample anchors per region (we'll use numpy and then move to torch)
    print("Sampling anchors per region:", anchors_per_region)
    anchors = []
    rng = np.random.RandomState(42)
    for i in range(1, 5):
        pts = sample_points_in_triangle_numpy(vertices[f"r{i}"], anchors_per_region)
        anchors.append(pts)

    # 3) initialize model
    model = SoftMinSoftmaxModel(anchors, device=device, use_region_affine=use_region_affine, use_sigmoid_transform=use_sigmoid_transform)
    model.to(device)

    # 4) prepare training dataset (grid points and targets). We'll convert to torch tensors and batch.
    X_all = torch.tensor(points_to_test.astype(np.float32), device=device)
    y_all = torch.tensor(membership_flat.astype(np.float32), device=device)

    # create dataloader style batching via random permutation each epoch
    N = X_all.shape[0]
    batch_size = max(64, N // batches_for_training)
    print(f"Training N={N}, batch_size={batch_size}, batches~{N//batch_size}")

    # 5) optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # small L2 regularization via weight decay could be added if needed
    mse_loss = nn.MSELoss()

    # Logging
    history = {"loss": []}

    # Training loop
    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        num_batches = 0
        for bstart in range(0, N, batch_size):
            bidx = perm[bstart:bstart + batch_size]
            xb = X_all[bidx]
            yb = y_all[bidx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().numpy())
            num_batches += 1
        epoch_loss /= num_batches
        history["loss"].append(epoch_loss)

        if (epoch % 50) == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:4d}/{n_epochs-1}  MSE={epoch_loss:.6f}  log_beta={model.log_beta_anchor.item():.4f}  log_gamma={model.log_gamma.item():.4f}")
        # early-stop-ish simple check (if loss extremely small)
        if epoch_loss < 1e-6:
            print("Very small loss reached; stopping early.")
            break

    # Optional polishing with LBFGS for a few iterations (uncomment if desired)
    try:
        print("Polishing with L-BFGS (optional, small iterations)...")
        def closure():
            optimizer_lbfgs.zero_grad()
            preds = model(X_all)
            l = mse_loss(preds, y_all)
            l.backward()
            return l
        optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=20, lr=1.0, tolerance_grad=1e-7)
        optimizer_lbfgs.step(closure)
    except Exception as e:
        print("LBFGS skipped or failed:", e)

    # Save model parameters
    torch.save({
        "model_state": model.state_dict(),
        "anchors": anchors,
        "history": history,
        "grid": {
            "xs": xs, "ys": ys, "Xg": Xg, "Yg": Yg
        },
        "membership_grid": membership_grid
    }, os.path.join(save_dir, "fit_checkpoint.pt"))
    print("Saved checkpoint to", os.path.join(save_dir, "fit_checkpoint.pt"))

    # Evaluate full grid predictions in batches and produce diagnostics
    model.eval()
    preds = []
    with torch.no_grad():
        B = 4096
        for s in range(0, N, B):
            xb = X_all[s:s+B]
            p = model(xb).detach().cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds, axis=0)
    pred_grid = preds.reshape(Xg.shape)
    diff = pred_grid - membership_grid

    # Save plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axs[0].imshow(membership_grid, origin='lower', extent=(xlim[0], xlim[1], ylim[0], ylim[1]), aspect='auto')
    axs[0].set_title('Monte-Carlo target')
    plt.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(pred_grid, origin='lower', extent=(xlim[0], xlim[1], ylim[0], ylim[1]), aspect='auto')
    axs[1].set_title('Model prediction')
    plt.colorbar(im1, ax=axs[1])
    im2 = axs[2].imshow(diff, origin='lower', extent=(xlim[0], xlim[1], ylim[0], ylim[1]), aspect='auto', cmap='RdBu')
    axs[2].set_title('Difference (pred - target)')
    plt.colorbar(im2, ax=axs[2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=150)
    print("Saved comparison plot to", os.path.join(save_dir, "comparison.png"))

    # Print final parameter values
    print("Final parameters:")
    print(" beta_anchor =", float(torch.exp(model.log_beta_anchor).item()))
    print(" gamma       =", float(torch.exp(model.log_gamma).item()))
    if use_region_affine:
        print(" region_scale:", model.region_scale.detach().cpu().numpy())
        print(" region_offset:", model.region_offset.detach().cpu().numpy())
    if use_sigmoid_transform:
        print(" out_scale, out_shift, out_sig_a:", model.out_scale.item(), model.out_shift.item(), model.out_sig_a.item())

    # return useful items
    return {
        "model": model,
        "anchors": anchors,
        "membership_grid": membership_grid,
        "pred_grid": pred_grid,
        "diff": diff,
        "history": history
    }

# --------------------------
# Main: argument parsing and run
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute_target", action="store_true", help="Force recompute Monte Carlo membership")
    parser.add_argument("--n_mc_samples", type=int, default=2000, help="Monte Carlo samples per grid point (costly)")
    parser.add_argument("--anchors_per_region", type=int, default=400, help="Number of fixed anchors used per region")
    parser.add_argument("--batches_for_training", type=int, default=64, help="Rough number of batches for training")
    parser.add_argument("--n_epochs", type=int, default=800, help="Training epochs (Adam)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate")
    parser.add_argument("--use_region_affine", action="store_true", help="Enable per-region affine transform on softmin")
    parser.add_argument("--use_sigmoid_transform", action="store_true", help="Enable final sigmoid transform")
    parser.add_argument("--save_dir", type=str, default="fit_results", help="Directory to save results")
    args = parser.parse_args()

    fit_model(
        membership_path=os.path.join(args.save_dir, "membership.npy"),
        recompute_target=args.recompute_target,
        n_mc_samples=args.n_mc_samples,
        anchors_per_region=args.anchors_per_region,
        batches_for_training=args.batches_for_training,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        use_region_affine=args.use_region_affine,
        use_sigmoid_transform=args.use_sigmoid_transform,
        save_dir=args.save_dir
    )

