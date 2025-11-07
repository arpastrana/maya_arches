import os
from math import fabs

import yaml
from functools import partial

# plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# jax
import jax.numpy as jnp
from jax import jit
from jax import vmap
from jax import Array

# jax cem
from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure

from maya_arches import FIGURES

from maya_arches.datastructures import create_topology_from_arch
from maya_arches.arches import create_arch
from maya_arches.optimization import calculate_start_params
from maya_arches.optimization import minimize_thrust_fn
from maya_arches.optimization import constraint_position_support_fn
from maya_arches.optimization import constraint_position_fn
from maya_arches.optimization import solve_thrust_opt_min
from maya_arches.optimization import solve_thrust_opt_max

from maya_arches.arches import Arch
from minmax import find_arch_type


# ==========================================================================
# Define functions
# ==========================================================================


def pretty_matplotlib():
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=32)  # 24
    plt.rc('axes', linewidth=1.5, labelsize=36) # 22

    plt.rc('xtick', labelsize=28, direction="in")
    plt.rc('ytick', labelsize=28, direction="in")
    
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    # tick settings
    plt.rc('xtick.major', size=10, pad=4)
    plt.rc('xtick.minor', size=5, pad=4)
    plt.rc('ytick.major', size=10)
    plt.rc('ytick.minor', size=5)


def find_constraint_position_y_idx(arch: Arch, structure: EquilibriumStructure, tol: float = 1e-6) -> tuple[int, float]:
    """
    Find the index of the node that is the last node on the corbel before the wall.
    """
    # NOTE: This is possible because the nodes in the structure are sorted by JAX CEM
    # Otherwise, we would need to use an index_node mapping to get the block
    index = -1
    for key in structure.nodes:

        # The first node is box constrained
        if key == 0:
            continue

        # The last node (support) is constrained differently
        if key == len(structure.nodes) - 1:
            continue

        block = arch.blocks[key]
        plane_line = block.plane_line()
        start = plane_line.start
        end = plane_line.end

        if start.y > end.y:
            start, end = end, start

        if start.y <= tol:
            break

        ymin = start.y
        index += 1

    return index, ymin


def constraint_position_y_fn(params: Array, model: EquilibriumModel, structure: EquilibriumStructure, ymin: float, index: int) -> Array:
    """
    The constraint function to ensure the thrust network is within the arch.
    """
    # reassemble model
    y = constraint_position_fn(params, model, structure)[index]

    return y - ymin
    

# ==========================================================================
# Initial parameters
# ==========================================================================

name = "grid"

# parameters
px0 = -1.0

# loss function
loss_fn_idx = 0
loss_fns = {
    0: minimize_thrust_fn,
    1: constraint_position_support_fn,   
    2: constraint_position_y_fn,
}

# optimization
optimize = True
tol = 1e-6
tol_bounds = 1e-3
maxiter = 100

# design space grid
sample_grid = True

# grid parameters, if auto_grid is False then these parameters are used
auto_grid = False
px_grid_start = -0.2
px_grid_end = -1.05
y_grid_start = 4.0
y_grid_end = 5.5
xticks = [0.25, 0.5, 0.75, 1.0]
yticks = [4.0, 4.5, 5.0, 5.5]
vmin = 0.0
vmax = 0.2

num_x = 50
num_y = 50
norm = "linear"

plot_contours = True
plot_y_hlines = True

# plotting
save_plot = True
plot_extension = "png"

# ==========================================================================
# Sample design space
# ==========================================================================

# Load yaml file with hyperparameters
with open("minmax.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

arch_config = config["arch"]
arch_type = find_arch_type(arch_config)
arch = create_arch(arch_type, **arch_config)
weight = arch.weight()
print(arch)
print(f"Weight: {weight:.3f}")

# Instantiate a topology diagram
topology = create_topology_from_arch(arch, px0=px0)

# JAX CEM - form finding
structure = EquilibriumStructure.from_topology_diagram(topology)
model = EquilibriumModel.from_topology_diagram(topology)

# Calculate start parameters
# 0: Horizontal load, 1: 2D coordinates of the origin node
params0 = calculate_start_params(topology)

# ==========================================================================
# Optimization
# ==========================================================================

params_opt = {"min": None, "max": None}
result = solve_thrust_opt_min(
    arch,
    params0,
    model,
    structure,
    maxiter=maxiter,
    tol=tol,
    tol_bounds=tol_bounds)

params_opt["min"] = result.x

result = solve_thrust_opt_max(
    arch,
    params0,
    model,
    structure,
    maxiter=maxiter,
    tol=tol,
    tol_bounds=tol_bounds)

params_opt["max"] = result.x

for key, params in params_opt.items():
    print(f"{key}: px: {params[0]:.3f} | y: {params[1]:.3f}")        

# ==========================================================================
# Estimate grid bounds
# ==========================================================================

if auto_grid:
    px_grid_start = params_opt["min"][0]
    px_grid_end = params_opt["max"][0]
    px_grid_diff = fabs(px_grid_end - px_grid_start) / 4.0
    px_grid_start = min(-1e-6, px_grid_start + px_grid_diff)
    px_grid_end = px_grid_end - px_grid_diff

    vmax = round((-1.0 * px_grid_end / weight) / 0.1) * 0.1
    vmax = round(vmax, 1)

    y_grid_start = params_opt["max"][1]
    y_grid_end = params_opt["min"][1]
    y_grid_diff = fabs(y_grid_end - y_grid_start)
    y_grid_start = round(y_grid_start - y_grid_diff, 1)
    y_grid_end = round(y_grid_end + y_grid_diff, 1)

    xticks = None

    print(f"vmax: {vmax:.3f}")
    print(f"px_grid_start: {px_grid_start:.3f}, px_grid_end: {px_grid_end:.3f}")
    print(f"y_grid_start: {y_grid_start:.3f}, y_grid_end: {y_grid_end:.3f}")

# ==========================================================================
# Parameter grid
# ==========================================================================

u_space = jnp.linspace(px_grid_start, px_grid_end, num=num_x)
v_space = jnp.linspace(y_grid_start, y_grid_end, num=num_y)
u_grid, v_grid = jnp.meshgrid(u_space, v_space)

# ==========================================================================
# Sample design space
# ==========================================================================

losses_values = {}

for loss_fn_idx in loss_fns.keys():

    # Calculate loss function
    def loss_fn(params, *args, **kwargs):
        _loss_fn = loss_fns[loss_fn_idx]
        return _loss_fn(params, model, structure, *args, **kwargs)

    if loss_fn_idx == 2:
        index, ymin = find_constraint_position_y_idx(arch, structure)
        print(f"Index: {index}, Ymin: {ymin}")
        loss_fn = partial(loss_fn, ymin=ymin, index=index)

    # Warmup
    loss_fn = jit(loss_fn)
    loss = loss_fn(params0)    

    # Create a vectorized version of loss_fn that operates on a single coordinate pair
    def loss_at_point(u, v):
        return loss_fn(jnp.array([u, v]))

    # Vectorize over both dimensions
    vectorized_loss = vmap(vmap(loss_at_point, in_axes=(0, None)), in_axes=(None, 0))

    # Compute loss values for all points in the grid at once
    loss_values = vectorized_loss(u_space, v_space)

    print(f"\nIndex: {loss_fn_idx}, Min: {jnp.min(loss_values):.3f}, Max: {jnp.max(loss_values):.3f}")
    losses_values[loss_fn_idx] = loss_values

# ==========================================================================
# Loss landscape
# ==========================================================================

pretty_matplotlib()
fig, ax = plt.subplots(figsize=(8, 8))

# Plot optimal design
for key, params in params_opt.items():
    ax.scatter(
        params[0] * -1.0, 
        params[1] * 1.0,                
        marker="o", 
        s=100, 
        facecolor="pink", 
        edgecolor="black", 
        zorder=100)

c = ax.pcolor(
    u_grid * -1.0, 
    v_grid * 1.0, 
    losses_values[0] / weight, 
    cmap='YlGnBu_r',
    norm=norm,
    vmin=vmin,
    vmax=vmax,
    alpha=1.0)

if plot_contours:
    # fmt = {1: r"$x_n-x_n^{l}=0$", 2: r"$y_i-y_i^{l}=0$"}    
    # manual = {1: [(3.2, 10.2)], 2: [(2.3, 7.8)]}

    for _idx in loss_fns.keys():
        if _idx == 0:
            continue

        cs = ax.contour(
            u_grid * -1.0, 
            v_grid * 1.0, 
            losses_values[_idx],                    
            levels=[0.0,],
            norm=norm,
            colors="black",
            origin="lower",
            linewidths=1.2,
            vmin=vmin,
            vmax=vmax,
            linestyles="dashdot")
        
        # Add labels to the contour lines    
        # _manual = manual[_idx]
        # clabels = ax.clabel(cs, inline=False, fontsize=20, fmt=_fmt, manual=_manual)
        # _fmt = {cs.levels[0]: fmt[_idx]}
        # clabels = ax.clabel(cs, inline=False, fontsize=20, fmt=_fmt)
        # for clabel in clabels:
            # clabel.set_verticalalignment("bottom")

if plot_y_hlines:
    for key, params in params_opt.items():
        ax.axhline(y=params[1], color="black", linewidth=1.2, linestyle="dotted")

# Labels
ax.set_xlabel(r'$-p_1$')
ax.set_ylabel(r'$y_1$')

# Ticks
if xticks is not None:
    ax.set_xticks(xticks)
if yticks is not None:
    ax.set_yticks(yticks)

# Set aspect ratio to be equal to ensure a square plot
# ax.set_aspect('equal', 'box')
ax.set_aspect('auto', 'box')

# Create a divider for the existing axes instance
divider = make_axes_locatable(ax)

# Append an axis on the right side of ax, for the colorbar
cax = divider.append_axes("right", size="3%", pad=0.2)

# Create colorbar
colorbar = fig.colorbar(c, cax=cax, orientation="vertical", extend=None)

# Set colorbar ticks, tick labels, and label
colorbar.set_ticks([vmin, vmax])
colorbar.set_ticklabels([vmin, vmax])
colorbar.set_label(r"$\tau$", labelpad=0, loc="center", rotation=0)
# colorbar.set_label(r"Thrust, $\tau$", labelpad=0, loc="center", rotation=90)

# Set title
# ax.set_title(fr"$\tau$", pad=20)

# Save figure
if save_plot:
    fig_name = f"{name}_{arch_config['type']}_h{int(arch.height)}_w{int(arch.width)}_wh{int(arch.wall_height * 10.0)}_ww{int(arch.wall_width * 10.0)}_lh{int(arch.lintel_height * 10.0)}_n{int(arch.num_blocks)}"
    filename = os.path.join(FIGURES, f"{fig_name}.{plot_extension}")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=300)
    print(f"Saved image to {filename}")

plt.tight_layout()
plt.show()
