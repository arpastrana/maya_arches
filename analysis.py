from functools import partial

from slicing import slice_vault
from slicing import create_slice_planes

from vaults import HalfMayanVault2D

from compas.colors import Color
from compas.geometry import add_vectors
from compas.utilities import pairwise

from compas_plotters import Plotter

# compas cem
from compas_cem.diagrams import TopologyDiagram
from compas_cem.elements import Node
from compas_cem.elements import TrailEdge
from compas_cem.elements import DeviationEdge
from compas_cem.loads import NodeLoad
from compas_cem.supports import NodeSupport
from compas_cem.equilibrium import static_equilibrium

# jax
import jax
import jaxopt

from jax import value_and_grad
from equinox import filter_jit as jit

from jax import jacfwd

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

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
width = 6.0

wall_width = 2.0
wall_height = 5.0
lintel_height = 2.0

num_slices = 15

block_density = 1.0
px0 = -1.0
py0 = -1.0

minmax_thrust = 1  # 0: minimize, 1: maximize
tol = 1e-6
maxiter = 100

show_loads = 0  # False

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

# ------------------------------------------------------------------------------
# Slicing
# ------------------------------------------------------------------------------

max_height = vault.height
if minmax_thrust:
    max_height = vault.wall_height + vault.corbel_height
    # max_height = vault.height - vault.height / num_slices
planes = create_slice_planes(vault, num_slices, max_height)
slices = slice_vault(vault, planes)

for i, slice in enumerate(slices):
    print(f"Slice {i}:\tLength:{slice.length:.2f}")
    assert slice.length <= vault.width / 2.0

# ------------------------------------------------------------------------------
# Instantiate a topology diagram
# ------------------------------------------------------------------------------

topology = TopologyDiagram()

# ------------------------------------------------------------------------------
# Add Nodes
# ------------------------------------------------------------------------------

node_keys = []
for i in range(num_slices + 1):
    factor = 1.0 - i / (num_slices)
    point = [factor * vault.width * 0.5, vault.height, 0.0]
    key = topology.add_node(Node(i, point))
    node_keys.append(key)

node_key_first = node_keys[0]
node_key_last = node_keys[-1]

# ------------------------------------------------------------------------------
# Set Supports Nodes
# ------------------------------------------------------------------------------

topology.add_support(NodeSupport(node_key_last))

# ------------------------------------------------------------------------------
# Add Loads
# ------------------------------------------------------------------------------

assert (len(node_keys) - 1) == len(slices), f"nodes: {len(node_keys) - 1} vs. {len(slices)}"

slices_reversed = slices[::-1]
node_keys_no_first = node_keys[1:]

for key, slice in zip(node_keys_no_first, slices_reversed):
    py = slice.length * block_density * -1.0
    load = [0.0, py, 0.0]
    topology.add_load(NodeLoad(key, load))


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


for i in range(num_slices):
    key = node_keys_no_first[i]

    if i == 0:
        slice_bot = slices_reversed[i]
        slice_top = slices_reversed[i]
        height = vault.lintel_height
    else:
        slice_bot = slices_reversed[i]
        slice_top = slices_reversed[i - 1]
        height = calculate_slices_height_delta(slice_bot, slice_top)

    slice_area = calculate_block_area(slice_bot, slice_top, height)
    print(i, "height", height, "area", slice_area)

    py = slice_area * block_density * -1.0
    load = [0.0, py, 0.0]
    topology.add_load(NodeLoad(key, load))

topology.add_load(NodeLoad(node_key_first, [px0, py0, 0.0]))

# ------------------------------------------------------------------------------
# Add Trail Edges
# ------------------------------------------------------------------------------

planes_reversed = planes[::-1]
for (u, v), plane in zip(pairwise(node_keys), planes_reversed):
    topology.add_edge(TrailEdge(u, v, length=-1.0, plane=plane))

# ------------------------------------------------------------------------------
# Build trails automatically
# ------------------------------------------------------------------------------

print()
topology.build_trails(auxiliary_trails=False)
print(topology)

# ------------------------------------------------------------------------------
# Compute a state of static equilibrium with COMPAS CEM
# ------------------------------------------------------------------------------

# form = static_equilibrium(topology, tmax=1)

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
# Functions
# ------------------------------------------------------------------------------

# Filter specification
filter_spec = jtu.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(
    lambda tree: (tree.loads, tree.xyz),
    filter_spec,
    replace=(True, True)
)


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

    # extract reaction force vector
    reaction_vector = eqstate.reactions[-1, :]
    assert reaction_vector.shape == (3,)

    # calculate error
    return jnp.sum(jnp.square(reaction_vector))


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
# Select loss function
# ------------------------------------------------------------------------------

if minmax_thrust == 0:
    print("\nMinimizing thrust")
    loss_fn = minimize_thrust_fn
elif minmax_thrust == 1:
    print("\nMaximizing thrust")
    loss_fn = maximize_thrust_fn
else:
    raise ValueError

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
# Loss, value and grad
# ------------------------------------------------------------------------------

print("\nCalculating initial loss and gradient values")
loss = loss_fn(params0, model)
value_and_grad_fn = jit(value_and_grad(loss_fn))
loss, gradient = value_and_grad_fn(params0, model)
print(f"Loss: {loss}")
print(f"Gradient: {gradient}")

# ------------------------------------------------------------------------------
# Bounds
# ------------------------------------------------------------------------------

print("\nGenerating box constraints")
load_bounds = [(None, None)]
xyz_bounds = [(vault.height - vault.lintel_height, vault.height)]
bounds = load_bounds + xyz_bounds

# ------------------------------------------------------------------------------
# Constraints
# ------------------------------------------------------------------------------

print("\nGenerating inequality constraints")
constraint_fn = jit(partial(constraint_fn, model=model))
constraint = constraint_fn(params0)
print(f"Constraint0: {constraint}")

lb = []
ub = []

slices_reversed = slices[::-1]
for i, slice in enumerate(slices_reversed):
    start = slice.start
    end = slice.end
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
    keep_feasible=False  # TODO: False
)

constraints = [constraint]

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

print("\nOptimizing")

optimizer_name = "SLSQP"
result = minimize(
    value_and_grad_fn,
    params0,
    model,
    method=optimizer_name,
    jac=True,
    bounds=bounds,
    constraints=constraints,
    tol=tol,
    options={"maxiter": maxiter},
    callback=None
)
print(result)

# generate optimized compas cem form diagram
params_star = result.x
model_star = rebuild_model_from_params(params_star, model)
eqstate_star = model_star(structure)
form_star = form_from_eqstate(structure, eqstate_star)

# ------------------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------------------

print()
for node in form_star.nodes():
    print(f"{node} X: {form_star.node_attribute(node, 'x'):.2f}\tPy: {form_star.node_attribute(node, 'qy'):.2f}")

print()
for edge in form_star.edges():
    print(f"{edge} Length: {form_star.edge_length(*edge):.2f}\tForce: {form_star.edge_force(edge):.2f}")


print(f"\nVertical load sum: {sum(form_star.node_attribute(node, 'qy') for node in form_star.nodes()):.2f}")

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

print("\nPlotting")
plotter = Plotter(figsize=(8, 8))

plotter.add(vault_polyline, linestyle="dashed", draw_points=False)

for slice in slices:
    plotter.add(
        slice,
        draw_as_segment=True,
        linestyle="solid",
        color=Color.purple()
    )

# plotter.add(topology)
# plotter.add(form0, show_reactions=False, show_loads=False)

plotter.add(
    form_star,
    show_nodes=True,
    show_reactions=False,
    nodesize=0.5,
    show_loads=show_loads,
    sizepolicy="relative",
)

plotter.zoom_extents()
plotter.show()
