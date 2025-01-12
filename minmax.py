"""
TODO: Plot minmax solutions in the same figure.
TODO: Automate script to sweep over various geometric ratios
TODO: Initialization strategy: minimize loadpath
"""
from functools import partial

from math import fabs

# from slicing import slice_vault
# from slicing import create_slice_planes
# from slicing import create_slice_planes_by_block

from vaults import HalfMayanVault2D

from compas.geometry import Plane
from compas.geometry import Reflection
from compas.geometry import Polyline
from compas.colors import Color
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.geometry import Point
from compas.geometry import Line
from compas.utilities import pairwise

from compas_plotters import Plotter

# compas cem
from compas_cem.diagrams import TopologyDiagram
from compas_cem.elements import Node
from compas_cem.elements import TrailEdge
from compas_cem.loads import NodeLoad
from compas_cem.supports import NodeSupport
from compas_cem.equilibrium import static_equilibrium

# jax
from jax import value_and_grad
from equinox import filter_jit as jit

from jax import jacfwd

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

# jax cem
from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure
from jax_cem.equilibrium import form_from_eqstate


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

xyz_tol = 1e-3  # origin node height tolerance on bounds for numerical stability (no zero length segments)
tol = 1e-6  # tolerance for optimization
maxiter = 100  # maximum number of iterations

plot_loads = True
plot_thrusts = False
forcescale = 0.5
plot_constraints = True
constraint_plot_tol = 1e-6
plot_other_half = True
plot_edges_as_segments = True
save_plot = True

# ------------------------------------------------------------------------------
# Optimization functions
# ------------------------------------------------------------------------------

def rebuild_model_from_params(params, model):
    """
    """
    # unpack parameters
    px = params[0]
    y = params[1]

    # update arrays in place
    loads = model.loads.at[0, 0:1].set(px)
    xyz = model.xyz.at[0, 1:2].set(y)

    # update model pytree with equinox voodoo
    model = eqx.tree_at(
        lambda tree: (tree.loads, tree.xyz),
        model,
        replace=(loads, xyz)
    )

    return model


def minimize_thrust_fn(params, model):
    """
    The loss function
    """
    # reassemble model
    model = rebuild_model_from_params(params, model)

    # calculate equilibrium state
    eqstate = model(structure, tmax=1)  # TODO: tmax = 100?

    # extract horizontal reaction force vector
    reaction_vector = eqstate.reactions[-1, :1]
    assert reaction_vector.shape == (1,)

    # calculate error
    return jnp.sqrt(jnp.sum(jnp.square(reaction_vector)))


def maximize_thrust_fn(params, model):
    """
    The loss function
    """
    return -1.0 * minimize_thrust_fn(params, model)


def constraint_fn(params, model):
    """
    """
    # reassemble model
    model = rebuild_model_from_params(params, model)

    # calculate equilibrium state
    eqstate = model(structure, tmax=1)  # TODO: tmax = 100?

    # extract x coordinates
    x = eqstate.xyz[:, 0]

    # select nodes of interest
    x = x[1:]

    return x


# ------------------------------------------------------------------------------
# Block functions
# ------------------------------------------------------------------------------

def calculate_block_area(slice_bottom, slice_top, height):
    """
    """
    base = slice_bottom.length + slice_top.length

    return base * height / 2.0


def calculate_slices_height_delta(slice_bottom, slice_top):
    """
    """
    yb = slice_bottom.midpoint.y
    yt = slice_top.midpoint.y

    return yt - yb

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

vault_polyline = vault.polyline()
vault_polygon = vault.polygon()

vault.blockify(num_blocks, slicing_method)

# ------------------------------------------------------------------------------
# Instantiate a topology diagram
# ------------------------------------------------------------------------------

topology = TopologyDiagram()

# ------------------------------------------------------------------------------
# Add Nodes
# ------------------------------------------------------------------------------

node_keys = []
for i in range(num_blocks + 1):
    factor = 1.0 - i / (num_blocks)
    point = [factor * vault.width * 0.5, vault.height, 0.0]
    key = topology.add_node(Node(i, point))
    node_keys.append(key)

node_key_first = node_keys[0]
node_key_last = node_keys[-1]
node_keys_reversed = node_keys[::-1]

# ------------------------------------------------------------------------------
# Set Supports Nodes
# ------------------------------------------------------------------------------

topology.add_support(NodeSupport(node_key_last))

# ------------------------------------------------------------------------------
# Add Loads
# ------------------------------------------------------------------------------

assert (len(node_keys) - 1) == len(vault.blocks), f"nodes: {len(node_keys) - 1} vs. {len(vault.blocks)}"

node_keys_no_first = node_keys[1:]
node_keys_no_first_reversed = node_keys_no_first[::-1]

# Apply loads to all nodes except the first
for key, block in zip(node_keys_no_first_reversed, vault.blocks):
    py = block.weight(block_density * -1.0)
    load = [0.0, py, 0.0]
    topology.add_load(NodeLoad(key, load))

# Modify the first and second node loads
block_last = vault.blocks[-1]  # The last block is the lintel roof
py0 = block_last.weight(block_density * -1.0) / 2.0
for key in node_keys[:2]:
    topology.add_load(NodeLoad(key, [px0, py0, 0.0]))

# ------------------------------------------------------------------------------
# Add Trail Edges
# ------------------------------------------------------------------------------

for (u, v), block in zip(pairwise(node_keys_reversed), vault.blocks):
    topology.add_edge(TrailEdge(u, v, length=-1.0, plane=block.plane))

# ------------------------------------------------------------------------------
# Build trails automatically
# ------------------------------------------------------------------------------

print()
topology.build_trails(auxiliary_trails=False)
print(topology)

# ------------------------------------------------------------------------------
# Compute a state of static equilibrium with COMPAS CEM
# ------------------------------------------------------------------------------

form_compas = static_equilibrium(topology, tmax=1)

# ------------------------------------------------------------------------------
# JAX CEM - form finding
# ------------------------------------------------------------------------------

structure = EquilibriumStructure.from_topology_diagram(topology)
model = EquilibriumModel.from_topology_diagram(topology)

eqstate = model(structure)
reaction = eqstate.reactions[node_key_last, :]
form0 = form_from_eqstate(structure, eqstate)

print("\nLoads")
for load in eqstate.loads:
    print(load)

print("\nXYZ")
for xyz in eqstate.xyz:
    print(xyz)

print("\nReactions")
for reaction in eqstate.reactions:
    print(reaction)

# ------------------------------------------------------------------------------
# Filter specification
# ------------------------------------------------------------------------------

filter_spec = jtu.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(
    lambda tree: (tree.loads, tree.xyz),
    filter_spec,
    replace=(True, True)
)

# ------------------------------------------------------------------------------
# Start parameters
# ------------------------------------------------------------------------------

print("\nExtracting initial parameters")

params0 = [
    topology.node_load(node_key_first)[0],
    topology.node_coordinates(node_key_first)[1]
]
params0 = jnp.array(params0)
print(f"{params0=}")
assert params0.size == 2

# ------------------------------------------------------------------------------
# Bounds
# ------------------------------------------------------------------------------

print("\nGenerating box constraints")

load_bounds = [(None, None)]
y_bounds = [(vault.height - vault.lintel_height + xyz_tol, vault.height - xyz_tol)]
bounds = load_bounds + y_bounds

# ------------------------------------------------------------------------------
# Constraints
# ------------------------------------------------------------------------------

print("\nGenerating inequality constraints")

constraint_fn = jit(partial(constraint_fn, model=model))
constraint = constraint_fn(params0)
print(f"Constraint0: {constraint}")

lb = []
ub = []

for block in vault.blocks[::-1]:
    start = block.line_bottom.start
    end = block.line_bottom.end
    lb.append(start.x)
    ub.append(end.x)

lb = jnp.array(lb)
ub = jnp.array(ub)

jac_fn = jit(jacfwd(constraint_fn))
jac = jac_fn(params0)

constraint = NonlinearConstraint(
    fun=constraint_fn,
    lb=lb,
    ub=ub,
    jac=jac_fn,
    keep_feasible=False
)

constraints = [constraint]

# ------------------------------------------------------------------------------
# Loss, value and grad
# ------------------------------------------------------------------------------

forms_star = {}
# loss_fns = {"min": minimize_thrust_fn, "max": maximize_thrust_fn}
loss_fns = {"min": minimize_thrust_fn}

for loss_fn_name, loss_fn in loss_fns.items():
    print(f"\nCalculating initial loss and gradient values for {loss_fn_name} solution")

    loss = loss_fn(params0, model)
    value_and_grad_fn = jit(value_and_grad(loss_fn))
    loss, gradient = value_and_grad_fn(params0, model)

    print(f"Loss: {loss}")
    print(f"Gradient: {gradient}")

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

    print("\nOptimizing")

    result = minimize(
        value_and_grad_fn,
        params0,
        model,
        method="SLSQP",
        jac=True,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        options={"maxiter": maxiter},
        callback=None
    )

    print(result)

    # Generate optimized compas cem form diagram
    params_star = result.x
    model_star = rebuild_model_from_params(params_star, model)
    eqstate_star = model_star(structure)
    form_star = form_from_eqstate(structure, eqstate_star)
    forms_star[loss_fn_name] = form_star

# ------------------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------------------

    print()
    for node in form_star.nodes():
        print(f"{node} X: {form_star.node_attribute(node, 'x'):.2f}\tPy: {form_star.node_attribute(node, 'qy'):.2f}")

    print()
    for edge in form_star.edges():
        print(f"{edge} Length: {form_star.edge_length(*edge):.2f}\tForce: {form_star.edge_force(edge):.2f}")

    sw = sum(fabs(form_star.node_attribute(node, 'qy')) for node in form_star.nodes())
    thrust = fabs(result.fun)
    print(f"\nVertical load sum (SW): {sw:.2f}")
    print(f"\nThrust: {thrust:.4f}")
    print(f"\nRatio thrust / SW [%]: {100.0 * thrust / sw:.1f}")

# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

print("\nStarting plotter")
plotter = Plotter(figsize=(8, 8))

plotter.add(vault_polyline, linestyle="solid", lineweight=2.0, draw_points=False)

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
        vault_polygon.transformed(R),
        linewidth=0.0,
        facecolor=Color.from_rgb255(240, 240, 240),
        zorder=50
        )

# ------------------------------------------------------------------------------
# Plot before zooming
# ------------------------------------------------------------------------------

color_blue = Color.from_rgb255(12, 119, 184)

for loss_fn_name, form_star in forms_star.items():

    print(f"\nAdding form to plotter for {loss_fn_name}")

    if plot_edges_as_segments:
        linestyle = "solid"
        if loss_fn_name == "min":
            linestyle = "dashed"

        _polyline = Polyline([form_star.node_coordinates(node) for node in node_keys])      
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
        print(f"\nPlotting constraints")
        assert len(node_keys_no_first) == len(vault.blocks), f"nodes: {len(node_keys_no_first)} vs. {len(vault.blocks)}"
      
        color_constraint_extrados = Color.from_rgb255(250, 80, 210)
        color_constraint_intrados = Color.orange()

        for node, block in zip(node_keys_no_first[::-1], vault.blocks):
            slice = block.line_bottom
            x, y, z = form_star.node_coordinates(node)

            # Check extrados
            x_constraint = slice.start.x
            if x - constraint_plot_tol <= x_constraint:            
                print(f"\tNode {node} for {loss_fn_name} is on the extrados")
                point = Point(x, y, z)
                plotter.add(point, size=6.0, facecolor=color_constraint_extrados, zorder=2000)

            # Check intrados
            x_constraint = slice.end.x
            if x + constraint_plot_tol >= x_constraint:
                print(f"\tNode {node} for {loss_fn_name} is on the intrados")
                point = Point(x, y, z)
                plotter.add(point, size=6.0, facecolor=color_constraint_intrados, zorder=2000)

        x, y, z = form_star.node_coordinates(node_key_first)
        if y - constraint_plot_tol <= y_bounds[0][0]:
            print(f"\tNode {node_key_first} for {loss_fn_name} is on the lower bound")
            point = Point(x, y, z)
            plotter.add(point, size=6.0, facecolor=color_constraint_intrados, zorder=2000)

        if y + constraint_plot_tol >= y_bounds[0][1]:
            print(f"\tNode {node_key_first} for {loss_fn_name} is on the upper bound")
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
