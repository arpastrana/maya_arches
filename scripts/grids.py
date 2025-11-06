import os
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

from minmax import find_arch_type
from maya_arches.arches import Arch


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
loss_fn_idx = 2
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
px_grid_start = -1.0
px_grid_end = -4.0
y_grid_start = 8.0
y_grid_end = 11.0

num_x = 50
num_y = 50
# vmin = 1.0
# vmax = 1.5
norm = "linear"

plot_y_hlines = True

# plotting
save_plot = False
plot_extension = "png"

# ==========================================================================
# Sample design space
# ==========================================================================

# Load yaml file with hyperparameters
with open("minmax.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

arch_type = find_arch_type(config["arch"])
arch = create_arch(arch_type, **config["arch"])
print(arch)

# Instantiate a topology diagram
topology = create_topology_from_arch(arch, px0=px0)

# JAX CEM - form finding
structure = EquilibriumStructure.from_topology_diagram(topology)
model = EquilibriumModel.from_topology_diagram(topology)

# Calculate start parameters
# 0: Horizontal load, 1: 2D coordinates of the origin node
params0 = calculate_start_params(topology)

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
print(f"Loss: {loss}")

# Optimization
if optimize:
    params_opt = []
    result = solve_thrust_opt_min(
        arch,
        params0,
        model,
        structure,
        maxiter=maxiter,
        tol=tol,
        tol_bounds=tol_bounds)

    params_opt.append(result.x)

    result = solve_thrust_opt_max(
        arch,
        params0,
        model,
        structure,
        maxiter=maxiter,
        tol=tol,
        tol_bounds=tol_bounds)

    params_opt.append(result.x)

    for params in params_opt:
        print(f"Thrust: {params[0]:.3f} | Position: {params[1]:.3f} | Loss: {loss_fn(params):.3f}")

# ==========================================================================
# Sample design space
# ==========================================================================

pretty_matplotlib()

# parameter grid
u_space = jnp.linspace(px_grid_start + 0.5, px_grid_end - 0.5, num=num_x)
v_space = jnp.linspace(y_grid_start - 0.5, y_grid_end + 0.5, num=num_y)
u_grid, v_grid = jnp.meshgrid(u_space, v_space)

# Create a vectorized version of loss_fn that operates on a single coordinate pair
def loss_at_point(u, v):
    return loss_fn(jnp.array([u, v]))

# Vectorize over both dimensions
vectorized_loss = vmap(vmap(loss_at_point, in_axes=(0, None)), in_axes=(None, 0))

# Compute loss values for all points in the grid at once
loss_values = vectorized_loss(u_space, v_space)

print(f"\nMin: {jnp.min(loss_values):.3f}, Max: {jnp.max(loss_values):.3f}")

# ==========================================================================
# Loss landscape
# ==========================================================================

fig, ax = plt.subplots(figsize=(8, 8))

# Plot optimal design
if optimize:
    markers = ["o", "o"]    
    for params, marker in zip(params_opt, markers):
        ax.scatter(
            params[0] * -1.0, 
            params[1] * 1.0,                
            marker=marker, 
            s=100, 
            facecolor="orange", 
            edgecolor="black", 
            zorder=100)

c = ax.pcolor(
    u_grid * -1.0, 
    v_grid * 1.0, 
    loss_values, 
    cmap='YlGnBu_r',
    norm=norm,
    # vmin=vmin,
    # vmax=vmax,
    alpha=1.0)

if loss_fn_idx == 1 or loss_fn_idx == 2:
    cs = ax.contour(
        u_grid * -1.0, 
        v_grid * 1.0, 
        loss_values,                    
        levels=[0.0,],
        norm=norm,
        colors="black",
        origin="lower",
        linewidths=0.5,
        linestyles="dashdot",
        # vmin=vmin,
        # vmax=vmax
        )
    
    # Add labels to the contour lines
    ax.clabel(cs, inline=True, fontsize=20, fmt=r"%1.1f")

if plot_y_hlines:
    ax.axhline(y=9.0, color="black", linewidth=0.5, linestyle="--")
    ax.axhline(y=10.0, color="black", linewidth=0.5, linestyle="--")        

# Labels
if loss_fn_idx == 0:
    ax.set_xlabel(r'$p_x$')
else:
    ax.set_xlabel(r'$t_x$')

ax.set_ylabel(r'$y_1$')

# Ticks
ax.set_xticks([1.0, 2.0, 3.0, 4.0])
ax.set_yticks([8.0, 9.0, 10.0, 11.0])

# Set aspect ratio to be equal to ensure a square plot
ax.set_aspect('equal', 'box')

# Create a divider for the existing axes instance
divider = make_axes_locatable(ax)

# Append an axis on the right side of ax, for the colorbar
cax = divider.append_axes("right", size="3%", pad=0.2)

# Create colorbar
colorbar = fig.colorbar(c, cax=cax, orientation="vertical", extend=None)

# Set colorbar ticks, tick labels, and label
# colorbar.set_ticks([vmin, vmax])
# colorbar.set_ticklabels([vmin, vmax])
# colorbar.set_label(r"Energy ratio, $\Omega / \Omega^{\star}$", labelpad=-25)
labels = {
    0: r"Thrust",
    1: r"Constraint 1",
    2: r"Constraint 2",
}
# colorbar.set_label(labels[loss_fn_idx], labelpad=25)

# Save figure
if save_plot:
    filename = os.path.join(FIGURES, f"{name}_{loss_fn_idx}.{plot_extension}")        
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=300)
    print(f"Saved image to {filename}")

plt.tight_layout()
plt.show()
