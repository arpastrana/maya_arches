"""
TODO: Automate script to sweep over various geometric ratios
TODO: Initialization strategy: minimize loadpath
"""
import os
from warnings import warn

# jax cem
from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure

# vaults
from mayan_vaults.vaults import HalfMayanVault2D

from mayan_vaults.datastructures import create_topology_from_vault

from mayan_vaults.optimization import solve_thrust_min
from mayan_vaults.optimization import solve_thrust_max
from mayan_vaults.optimization import calculate_start_params
from mayan_vaults.optimization import create_thrust_network_from_opt_result
from mayan_vaults.optimization import test_thrust_opt_result

from mayan_vaults.plotting import VaultPlotter

from mayan_vaults import FIGURES

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

height = 10.0
width = 8.0  # we only consider half this value for symmetry

wall_width = 2.0
wall_height = 5.0
lintel_height = 1.0

num_blocks = 10
slicing_method = 0  # 0: block, 1: uniform height
block_density = 1.0
px0 = -1.0  # initial guess for horizontal load at origin node (top)

tol_bounds = 1e-3  # origin node height tolerance on bounds for numerical stability (no zero length segments)
tol = 1e-6  # tolerance for optimization
maxiter = 100  # maximum number of iterations

forcescale = 0.5
plot_constraints = True
plot_other_half = True

plot_loads = False
plot_thrusts = False

save_plot = True

# ------------------------------------------------------------------------------
# Create a Mayan vault
# ------------------------------------------------------------------------------

vault = HalfMayanVault2D(
    height,
    width,
    wall_height,
    wall_width,
    lintel_height
)

vault.blockify(num_blocks, block_density, slicing_method)

# ------------------------------------------------------------------------------
# Instantiate a topology diagram
# ------------------------------------------------------------------------------

topology = create_topology_from_vault(vault, px0)

# ------------------------------------------------------------------------------
# JAX CEM - form finding
# ------------------------------------------------------------------------------

structure = EquilibriumStructure.from_topology_diagram(topology)
model = EquilibriumModel.from_topology_diagram(topology)

# ------------------------------------------------------------------------------
# Start parameters
# ------------------------------------------------------------------------------

# The parameters are the horizontal load and the 2D coordinates of the origin node
params0 = calculate_start_params(topology)

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

networks = {}
solve_fns = {"min": solve_thrust_min, "max": solve_thrust_max}

for solve_fn_name, solve_fn in solve_fns.items():

    print(f"\n***** Solving for {solve_fn_name} solution *****\n")
    result = solve_fn(
        vault,
        params0,
        model,
        structure,
        maxiter=maxiter,
        tol=tol,
        tol_bounds=tol_bounds
    )

    if not result.success:
        warn("Optimization failed\n")
        print(result)

# ------------------------------------------------------------------------------
# Generate thrust network
# ------------------------------------------------------------------------------

    network = create_thrust_network_from_opt_result(result, model, structure)

# ------------------------------------------------------------------------------
# Check results
# ------------------------------------------------------------------------------

    test_thrust_opt_result(network, vault, result)
    networks[solve_fn_name] = network

# ------------------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------------------

    sw = vault.weight()
    thrust = network.thrust()
    print(f"SW (Vertical load sum): {sw:.2f}")
    print(f"Thrust at support: {thrust:.2f}")
    print(f"Ratio thrust / SW [%]: {100.0 * thrust / sw:.1f}")

# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

print("\n***** Plotting *****\n")

plotter = VaultPlotter(figsize=(8, 8))
plotter.plot_vault(vault, plot_other_half)
plotter.plot_vault_blocks(vault)

for loss_fn_name, network in networks.items():    
    linestyle = "solid" if loss_fn_name == "max" else "dashed"
    plotter.plot_thrust_network(network, linestyle=linestyle)

    if plot_constraints:
        plotter.plot_constraints(vault, network, tol_bounds)

plotter.zoom_extents()

for loss_fn_name, network in networks.items():
    if plot_loads:
        plotter.plot_thrust_network_loads(network, forcescale)

    if plot_thrusts:
        plotter.plot_thrust_network_thrusts(network, forcescale)

if save_plot:
    fig_name = f"minmax_h{int(height)}_w{int(width)}_wh{int(wall_height)}_ww{int(wall_width)}_lh{int(lintel_height)}.pdf"
    fig_path = os.path.join(FIGURES, fig_name)
    plotter.save(fig_path, transparent=True, bbox_inches="tight")
    print(f"\nSaved figure to {fig_path}")

plotter.show()

