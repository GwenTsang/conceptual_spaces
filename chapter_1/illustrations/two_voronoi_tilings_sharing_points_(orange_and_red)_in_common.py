# -*- coding: utf-8 -*-
"""Two Voronoi tilings sharing points (orange and red) in common.ipynb

Original file is located at
    https://colab.research.google.com/drive/14mWU7dlup7_9lOkIEwYp7SasOmxnzxju
"""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from matplotlib.path import Path

# -------------------------
# Parameters
# -------------------------
num_points = 21
R = 10
np.random.seed(1)
target_cm = 6.5  # desired size for the largest dimension (between 6 and 7 cm)

# Dot sizes (in pt)
dot_black_pt = 6.0   # black dots
dot_orange_pt = 4.0  # adjacent points (orange)
dot_red_pt = 7.0     # central red dot

# Frame appearance
frame_line_width_pt = 1.8
frame_draw_opacity = 0.95

# Caption text (edit this string to change the caption)
caption_text = ("Two Voronoi diagrams. Left: original diagram (central red point, orange adjacent points, "
                "black non-adjacent). Right: diagram after adding black points outside the central--adjacent hull.")

# -------------------------
# Helper functions
# -------------------------
def fmt(x):
    return f"{x:.6f}"

def compute_voronoi_edges(points):
    """Return Voronoi object and list of edges (x1,y1,x2,y2) including extended infinite ridges."""
    vor = Voronoi(points)
    edges = []
    # finite ridges
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            v1 = vor.vertices[simplex[0]]
            v2 = vor.vertices[simplex[1]]
            edges.append((v1[0], v1[1], v2[0], v2[1]))
    # infinite ridges: extend similarly to your original code
    ptp_bound = np.ptp(points, axis=0)
    max_bound = ptp_bound.max()
    for (p1_idx, p2_idx), simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            p1 = vor.points[p1_idx]
            p2 = vor.points[p2_idx]
            t = p2 - p1
            norm = np.linalg.norm(t)
            if norm == 0:
                continue
            t = t / norm
            n = np.array([-t[1], t[0]])
            finite_vertex_idx = simplex[simplex >= 0][0]
            finite_vertex = vor.vertices[finite_vertex_idx]
            midpoint = (p1 + p2) / 2
            direction = np.sign(np.dot(midpoint - points.mean(axis=0), n)) * n
            far_point = finite_vertex + direction * max_bound * 10.0
            edges.append((finite_vertex[0], finite_vertex[1], far_point[0], far_point[1]))
    return vor, edges

# -------------------------
# Generate original points (same as your code)
# -------------------------
theta = np.random.uniform(0, 2 * np.pi, num_points)
r = R * np.sqrt(np.random.uniform(0, 1, num_points))
x = r * np.cos(theta)
y = r * np.sin(theta)
other_points = np.column_stack((x, y))
central_point = np.array([[0.0, 0.0]])
points = np.vstack((central_point, other_points))

# Compute adjacency (neighbors of central point 0)
vor_full = Voronoi(points)
ridge_points = vor_full.ridge_points
Adj_indices = []
for pair in ridge_points:
    if 0 in pair:
        neighbor = pair[0] if pair[1] == 0 else pair[1]
        Adj_indices.append(neighbor)
Adj_indices = np.unique(Adj_indices)
Adj_indices = Adj_indices[Adj_indices != 0].astype(int)
Adj_P = points[Adj_indices.astype(int)]

# Build hull and hull_path used for sampling new black points
hull_points = np.vstack((central_point, Adj_P))
hull = ConvexHull(hull_points)
hull_path = Path(hull_points[hull.vertices])

# Compute plotting range just like your original code
all_x = points[:, 0]
all_y = points[:, 1]
x_range = all_x.max() - all_x.min()
y_range = all_y.max() - all_y.min()
expansion_factor = 1.5
x_min_initial = all_x.min() - ((expansion_factor - 1) * x_range / 2)
x_max_initial = all_x.max() + ((expansion_factor - 1) * x_range / 2)
y_min_initial = all_y.min() - ((expansion_factor - 1) * y_range / 2)
y_max_initial = all_y.max() + ((expansion_factor - 1) * y_range / 2)
xlim = (x_min_initial, x_max_initial)
ylim = (y_min_initial, y_max_initial)

# -------------------------
# First Voronoi (original points)
# -------------------------
vor1, edges1 = compute_voronoi_edges(points)
central_idx1 = 0
all_indices = np.arange(len(points))
non_adj_indices1 = np.setdiff1d(all_indices, np.concatenate(([0], Adj_indices)))

# -------------------------
# Generate new black points for the second plot (same logic as your original script)
# -------------------------
min_distance_threshold = 1.0
max_attempts = 1000
attempt = 0
new_black_points = []
combined_adj_points = np.vstack((central_point, Adj_P))
num_black = len(np.setdiff1d(all_indices, np.concatenate(([0], Adj_indices))))

while len(new_black_points) < num_black and attempt < max_attempts * num_black:
    x_candidate = np.random.uniform(x_min_initial, x_max_initial)
    y_candidate = np.random.uniform(y_min_initial, y_max_initial)
    candidate_point = np.array([x_candidate, y_candidate])
    # must lie outside the hull
    if not hull_path.contains_point(candidate_point):
        distances = np.linalg.norm(combined_adj_points - candidate_point, axis=1)
        if np.min(distances) >= min_distance_threshold:
            new_black_points.append(candidate_point)
    attempt += 1

new_black_points = np.array(new_black_points)

# Build second set of points: central + Adj_P + new_black_points
points2 = np.vstack((central_point, Adj_P, new_black_points))
vor2, edges2 = compute_voronoi_edges(points2)

# For the second diagram: indices
central_idx2 = 0
adj_count = len(Adj_P)
adj_indices2 = np.arange(1, adj_count + 1)  # 1..adj_count
black_indices2 = np.arange(adj_count + 1, len(points2))  # rest

# -------------------------
# Compute overall scale factor so the largest dimension (same for both) becomes target_cm
# -------------------------
width_data = xlim[1] - xlim[0]
height_data = ylim[1] - ylim[0]
max_dim_data = max(width_data, height_data)
if max_dim_data <= 0:
    scale_factor = 1.0
else:
    scale_factor = target_cm / max_dim_data

# -------------------------
# Build LaTeX lines (two tikz pictures side-by-side inside the template) + caption
# -------------------------
lines = []
lines.append(r"\documentclass{article}")
lines.append("")
lines.append(r"\usepackage{tikz}")
lines.append(r"\usepackage{xcolor}")  # ensure orange color is available
lines.append(r"\usepackage{graphicx} % --- We import the package for \scalebox---")
lines.append(r"\usepackage[margin=1in]{geometry}")
lines.append("")
lines.append(r"\begin{document}")
lines.append("")
lines.append(r"\begin{figure}[htbp]")
lines.append(r"\centering")
lines.append("")

# First picture (left)
lines.append(f"\\scalebox{{{scale_factor:.8f}}}{{%")
lines.append(r"  \begin{tikzpicture}[x=1cm,y=1cm] % <-- coordinates in cm; scaled by \scalebox")
lines.append("")
lines.append(f"    \\clip ({fmt(xlim[0])},{fmt(ylim[0])}) rectangle ({fmt(xlim[1])},{fmt(ylim[1])});")
lines.append("")
lines.append("    % Voronoi edges (first diagram)")
for (x1, y1, x2, y2) in edges1:
    lines.append(f"    \\draw[black, line width=0.6pt] ({fmt(x1)},{fmt(y1)}) -- ({fmt(x2)},{fmt(y2)});")
lines.append("")
lines.append("    % Non-adjacent points (black, larger) - first diagram")
for idx in non_adj_indices1:
    px, py = points[int(idx)]
    lines.append(f"    \\fill[black] ({fmt(px)},{fmt(py)}) circle ({dot_black_pt}pt);")
lines.append("")
lines.append("    % Adjacent points (orange, larger) - first diagram")
for idx in Adj_indices:
    px, py = points[int(idx)]
    lines.append(f"    \\fill[orange] ({fmt(px)},{fmt(py)}) circle ({dot_orange_pt}pt);")
lines.append("")
cx1, cy1 = points[central_idx1]
lines.append("    % Central point (red, larger) - first diagram")
lines.append(f"    \\fill[red] ({fmt(cx1)},{fmt(cy1)}) circle ({dot_red_pt}pt);")
lines.append("")
# Frame for first
lines.append("    % Black frame around the drawing (first)")
lines.append(
    f"    \\draw[black, line width={frame_line_width_pt}pt, draw opacity={frame_draw_opacity:.3f}] "
    f"({fmt(xlim[0])},{fmt(ylim[0])}) rectangle ({fmt(xlim[1])},{fmt(ylim[1])});"
)
lines.append("")
lines.append(r"  \end{tikzpicture}%")
lines.append("}")  # close scalebox for first

# small horizontal separation
lines.append(r"\hspace{0.8cm}")

# Second picture (right)
lines.append(f"\\scalebox{{{scale_factor:.8f}}}{{%")
lines.append(r"  \begin{tikzpicture}[x=1cm,y=1cm] % <-- coordinates in cm; scaled by \scalebox")
lines.append("")
lines.append(f"    \\clip ({fmt(xlim[0])},{fmt(ylim[0])}) rectangle ({fmt(xlim[1])},{fmt(ylim[1])});")
lines.append("")
lines.append("    % Voronoi edges (second diagram)")
for (x1, y1, x2, y2) in edges2:
    lines.append(f"    \\draw[black, line width=0.6pt] ({fmt(x1)},{fmt(y1)}) -- ({fmt(x2)},{fmt(y2)});")
lines.append("")
lines.append("    % Black points added for second diagram (larger)")
for idx in black_indices2:
    px, py = points2[int(idx)]
    lines.append(f"    \\fill[black] ({fmt(px)},{fmt(py)}) circle ({dot_black_pt}pt);")
lines.append("")
lines.append("    % Adjacent points (orange, larger) - second diagram")
for idx in adj_indices2:
    px, py = points2[int(idx)]
    lines.append(f"    \\fill[orange] ({fmt(px)},{fmt(py)}) circle ({dot_orange_pt}pt);")
lines.append("")
cx2, cy2 = points2[central_idx2]
lines.append("    % Central point (red, larger) - second diagram")
lines.append(f"    \\fill[red] ({fmt(cx2)},{fmt(cy2)}) circle ({dot_red_pt}pt);")
lines.append("")
# Frame for second
lines.append("    % Black frame around the drawing (second)")
lines.append(
    f"    \\draw[black, line width={frame_line_width_pt}pt, draw opacity={frame_draw_opacity:.3f}] "
    f"({fmt(xlim[0])},{fmt(ylim[0])}) rectangle ({fmt(xlim[1])},{fmt(ylim[1])});"
)
lines.append("")
lines.append(r"  \end{tikzpicture}%")
lines.append("}")  # close scalebox for second

# Add global caption under both images
lines.append("")
lines.append(f"\\caption{{{caption_text}}}")
lines.append("")

lines.append(r"\end{figure}")
lines.append("")
lines.append(r"\end{document}")

# -------------------------
# Write to file
# -------------------------
out_filename = "voronoi_two_scaled.tex"
with open(out_filename, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote TeX file to {out_filename}")
print(f"Scale factor used: {scale_factor:.8f} (largest dimension ≈ {target_cm} cm after scaling).")
print(f"Frame line width: {frame_line_width_pt} pt; frame opacity: {frame_draw_opacity:.3f}")
print("Dot sizes (pt): black=", dot_black_pt, " orange=", dot_orange_pt, " red=", dot_red_pt)
print("Caption written: ", caption_text)