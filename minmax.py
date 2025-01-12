"""
TODO: Plot minmax solutions in the same figure.
TODO: Automate script to sweep over various geometric ratios
TODO: Initialization strategy: minimize loadpath
"""
from functools import partial

from math import fabs

from vaults import HalfMayanVault2D

from diagrams import create_topology_from_vault

from optimization import solve_thrust_min
from optimization import solve_thrust_max
from optimization import calculate_start_params

from compas.geometry import Plane
from compas.geometry import Reflection
from compas.geometry import Polyline
from compas.colors import Color
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.geometry import Point
from compas.geometry import Line

from compas_plotters import Plotter

# jax cem
from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure


# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

height = 10.0
width = 8.0  # we only consider half this value for symmetry

wall_width = 2.0
wall_height = 5.0
lintel_height = 1.0

num_blocks = 10
slicing_method = 0  # 0: block, 1: uniform, by height
block_density = 1.0
px0 = -1.0  # initial guess for horizontal load at origin node (top)

tol_bounds = 1e-3  # origin node height tolerance on bounds for numerical stability (no zero length segments)
tol = 1e-6  # tolerance for optimization
maxiter = 100  # maximum number of iterations

plot_loads = False
plot_thrusts = False
forcescale = 0.5
plot_constraints = False
plot_other_half = True
plot_edges_as_segments = True
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

vault.blockify(num_blocks, slicing_method)

# ------------------------------------------------------------------------------
# Instantiate a topology diagram
# ------------------------------------------------------------------------------

topology = create_topology_from_vault(vault, block_density, px0)

# ------------------------------------------------------------------------------
# Compute a state of static equilibrium with COMPAS CEM
# ------------------------------------------------------------------------------

# form_compas = static_equilibrium(topology, tmax=1)

# ------------------------------------------------------------------------------
# JAX CEM - form finding
# ------------------------------------------------------------------------------

structure = EquilibriumStructure.from_topology_diagram(topology)
model = EquilibriumModel.from_topology_diagram(topology)

# eqstate = model(structure)
# form0 = form_from_eqstate(structure, eqstate)

# print("\nLoads")
# for load in eqstate.loads:
#     print(load)

# print("\nXYZ")
# for xyz in eqstate.xyz:
#     print(xyz)

# print("\nReactions")
# for reaction in eqstate.reactions:
#     print(reaction)

# ------------------------------------------------------------------------------
# Start parameters
# ------------------------------------------------------------------------------

print("\nExtracting starting parameters")
params0 = calculate_start_params(topology)

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

forms_star = {}
solve_fns = {"min": solve_thrust_min, "max": solve_thrust_max}

for solve_fn_name, solve_fn in solve_fns.items():

    print(f"\n***** Solving for {solve_fn_name} solution *****\n")
    form_star = solve_fn(
        vault,
        params0,
        model,
        structure,
        maxiter=maxiter,
        tol=tol,
        tol_bounds=tol_bounds
    )

    forms_star[solve_fn_name] = form_star

# ------------------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------------------

    print(f"Stats")
    sw = sum(fabs(form_star.node_attribute(node, 'qy')) for node in form_star.nodes())
    thrust = sum(fabs(form_star.node_attribute(node, 'rx')) for node in form_star.nodes() if form_star.is_node_support(node))
    print(f"\tSW (Vertical load sum): {sw:.2f}")
    print(f"\tThrust at support: {thrust:.2f}")
    print(f"\tRatio thrust / SW [%]: {100.0 * thrust / sw:.1f}")

# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

print("\nPlotting")
plotter = Plotter(figsize=(8, 8))

plotter.add(vault.polyline(), linestyle="solid", lineweight=2.0, draw_points=False)

for block in vault.blocks:
    plotter.add(
        block.line_bottom,
        draw_as_segment=True,
        linestyle="dotted",
        color=Color.grey(),
        lineweight=0.5,
        zorder=100
    )

if plot_other_half:
    plane = Plane((vault.width / 2.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    R = Reflection.from_plane(plane)
    plotter.add(
        vault.polygon().transformed(R),
        linewidth=0.0,
        facecolor=Color.from_rgb255(240, 240, 240),
        zorder=50
        )

# ------------------------------------------------------------------------------
# Plot before zooming
# ------------------------------------------------------------------------------

color_blue = Color.from_rgb255(12, 119, 184)

for loss_fn_name, form_star in forms_star.items():

    if plot_edges_as_segments:
        linestyle = "solid"
        if loss_fn_name == "min":
            linestyle = "dashed"

        _polyline = Polyline([form_star.node_coordinates(node) for node in range(form_star.number_of_nodes())])      
        plotter.add(
            _polyline,
            draw_points=False,
            linestyle=linestyle,
            color=color_blue,
            linewidth=3.0,
            zorder=1000
        )

    else:
        plotter.add(
            form_star,
            show_nodes=False,
            show_reactions=False,
            nodesize=0.75,
            show_loads=False,
            edgewidth=(2, 5),
            sizepolicy="relative",
        )

    if plot_constraints:
        assert len(node_keys_no_first) == len(vault.blocks), f"nodes: {len(node_keys_no_first)} vs. {len(vault.blocks)}"
      
        color_constraint_extrados = Color.from_rgb255(250, 80, 210)
        color_constraint_intrados = Color.orange()

        for node, block in zip(node_keys_no_first[::-1], vault.blocks):
            slice = block.line_bottom
            x, y, z = form_star.node_coordinates(node)

            # Check extrados
            x_constraint = slice.start.x
            if x - constraint_plot_tol <= x_constraint:                            
                point = Point(x, y, z)
                plotter.add(point, size=6.0, facecolor=color_constraint_extrados, zorder=2000)

            # Check intrados
            x_constraint = slice.end.x
            if x + constraint_plot_tol >= x_constraint:
                point = Point(x, y, z)
                plotter.add(point, size=6.0, facecolor=color_constraint_intrados, zorder=2000)

        x, y, z = form_star.node_coordinates(node_key_first)
        if y - constraint_plot_tol <= vault.height - vault.lintel_height + tol_bounds:
            point = Point(x, y, z)
            plotter.add(point, size=6.0, facecolor=color_constraint_intrados, zorder=2000)

        if y + constraint_plot_tol >= vault.height - tol_bounds:
            point = Point(x, y, z)
            plotter.add(point, size=6.0, facecolor=color_constraint_extrados, zorder=2000)

    if plot_loads:
        for node in form_star.nodes():
            load_y = [0.0, form_star.node_attribute(node, 'qy'), 0.0]
            xyz = form_star.node_coordinates(node)
            line = Line(xyz, add_vectors(xyz, scale_vector(load_y, forcescale)))
            plotter.add(
                line,
                draw_as_segment=True,
                linestyle="solid",
                color=Color.from_rgb255(0, 150, 10)
            )

plotter.zoom_extents()

# ------------------------------------------------------------------------------
# Plot thrusts
# ------------------------------------------------------------------------------

if plot_thrusts:
    for loss_fn_name, form_star in forms_star.items():
        _nodes = [node_key_first, node_key_last, node_key_last]
        load = [form_star.node_attribute(node_key_first, 'qx'), 0.0, 0.0]
        thrust_x = [form_star.node_attribute(node_key_last, 'rx'), 0.0, 0.0]
        thrust_y = [0.0, form_star.node_attribute(node_key_last, 'ry'), 0.0]
        _forces = [load, thrust_x, thrust_y]

        _nodes = [node_key_first, node_key_last]
        load = [form_star.node_attribute(node_key_first, 'qx'), 0.0, 0.0]
        thrust_x = [form_star.node_attribute(node_key_last, 'rx'), 0.0, 0.0]        
        _forces = [load, thrust_x]

        print()
        for node, force in zip(_nodes, _forces):
            print(f"Thrust: {force}")
            xyz = form_star.node_coordinates(node)
            line = Line(xyz, add_vectors(xyz, scale_vector(force, -1.0 * forcescale)))
            plotter.add(
                line,
                draw_as_segment=True,
                linestyle="solid",
                linewidth=2.0,
                color=Color.grey()
            )

if save_plot:
    plotter.save(f"figures/minmax.pdf", transparent=True, bbox_inches="tight")

plotter.show()
