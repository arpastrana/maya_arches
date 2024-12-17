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

from jax import grad
from equinox import filter_jit as jit

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

# jax cem
from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure
from jax_cem.equilibrium import form_from_eqstate


# ------------------------------------------------------------------------------
# Create a Mayan vault
# ------------------------------------------------------------------------------

height = 10.0
width = 8.0
wall_width = 2.0
wall_height = 5.0
lintel_height = 2.0
num_slices = 13

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

planes = create_slice_planes(vault, num_slices)
slices = slice_vault(vault, num_slices)

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
    print(point)
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

block_density = 1.0

# assert (len(node_keys) - 1) == len(slices), f"nodes: {len(node_keys) - 1} vs. {len(slices)}"

# slices_reversed = slices[::-1]
# for key, slice in zip(node_keys[:-1], slices_reversed):
    # py = slice.length * block_density * -1.0

px0 = -1.0
py0 = -2.0
topology.add_load(NodeLoad(node_key_first, [px0, py0, 0.0]))

for key in node_keys[1:]:
    py = -1.0
    load = [0.0, py, 0.0]
    topology.add_load(NodeLoad(key, load))

# ------------------------------------------------------------------------------
# Add Trail Edges
# ------------------------------------------------------------------------------

length = -1.0
planes_reversed = planes[::-1]
for (u, v), plane in zip(pairwise(node_keys), planes_reversed):
    topology.add_edge(TrailEdge(u, v, length=length, plane=plane))

# ------------------------------------------------------------------------------
# Add Deviation Edges
# ------------------------------------------------------------------------------

# force = -1.0
# xyz = add_vectors(
#     topology.node_coordinates(node_key_first),
#     [vault.width / 20.0, 0.0, 0.0]
# )
# node_key_extra = topology.add_node(Node(key=None, xyz=xyz))
# topology.add_edge(DeviationEdge(node_key_first, node_key_extra, force=force))

# ------------------------------------------------------------------------------
# Build trails automatically
# ------------------------------------------------------------------------------

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
form_jax = form_from_eqstate(structure, eqstate)

# for load in model.loads:
#     print(load)

# load = [-2.0, 0.0, 0.0]

# loads = model.loads.at[node_key_first, :].set(load)
# print()
# print("MODEL WITH LOADS")
# for load in model.loads:
#     print(load)

# print()
# print("LOADS NEW")
# for load in loads:
#     print(load)

# print("FILTER SPEC")
# print(filter_spec)

# diff_model, static_model = eqx.partition(model, filter_spec)
# print("DIFF MODEL 1")
# print(diff_model)

# print()
# model = eqx.tree_at(lambda tree: (tree.loads, ), filter_spec, replace=(loads, ))
# print("MODEL WITH LOADS?")
# for load in model.loads:
#     print(load)

# ------------------------------------------------------------------------------
# JAX CEM - optimization
# ------------------------------------------------------------------------------

# Filter specification
filter_spec = jtu.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(
    lambda tree: (tree.loads, tree.xyz),
    filter_spec,
    replace=(True, )
)


# define loss function
@jit
def loss_fn(params, model):
    """
    The loss function
    """
    # unpack parameters
    load = params[:3]
    xyz_origin = params[3:]

    # update arrays in place
    loads = model.loads.at[node_key_first, :].set(load)
    xyz = model.xyz.at[node_key_first, :].set(xyz_origin)

    # update model pytree with equinox voodoo
    model = eqx.tree_at(
        lambda tree: (tree.loads, tree.xyz),
        model,
        replace=(loads, xyz))

    # calculate equilibrium state
    eqstate = model(structure, tmax=1)  # TODO: tmax = 100?

    # extract reaction force vector
    reaction_vector = eqstate.residuals[node_key_last, :]
    assert reaction_vector.shape == (3,)

    # calculate error
    return jnp.sum(jnp.square(reaction_vector))

raise

# # define loss function
# @jit
# def loss_fn(diff_model, static_model):
#     """
#     The loss function
#     """
#     model = eqx.combine(diff_model, static_model)
#     eqstate = model(structure, tmax=100)

#     reaction_vector = eqstate.residuals[node_key_last, :]
#     assert reaction_vector.shape == (3,)

#     return jnp.sum(jnp.square(reaction_vector))


# # set tree filtering specification
# filter_spec = jtu.tree_map(lambda _: False, model)
# filter_spec = eqx.tree_at(lambda tree: (tree.lengths, tree.forces), filter_spec, replace=(True, True))

# # split model into differentiable and static submodels
# diff_model, static_model = eqx.partition(model, filter_spec)

# # create bounds
# bound_low = eqx.tree_at(lambda tree: (tree.lengths, tree.forces),
#                         diff_model,
#                         replace=(-3. * np.ones_like(model.forces), -3.0 * np.ones_like(model.lengths)))
# # print(bound_low.lengths)
# # print(bound_low.forces)
# bound_up = eqx.tree_at(lambda tree: (tree.lengths, tree.forces),
#                        diff_model,
#                        replace=(17. * np.ones_like(model.forces), 15. * np.ones_like(model.lengths)))
# # print(bound_up.lengths)
# # print(bound_up.forces)

# bounds = (bound_low, bound_up)

# # evaluate loss function at the start
# loss = loss_fn(diff_model, static_model, structure, y)
# print(f"{loss=}")

# # solve optimization problem with scipy
# print("\n***Optimizing with scipy***")
# # optimizer = jaxopt.ScipyMinimize
# optimizer = jaxopt.ScipyBoundedMinimize

# opt = optimizer(fun=loss_fn, method="L-BFGS-B", jit=True, tol=1e-6, maxiter=100)

# # opt_result = opt.run(diff_model, static_model, structure, y)
# opt_result = opt.run(diff_model, bounds, static_model, structure, y)
# diff_model_star, opt_state_star = opt_result

# # evaluate loss function at optimum point
# loss = loss_fn(diff_model_star, static_model, structure, y)
# print(f"{loss=}")
# # print(opt_state_star)

# # generate optimized compas cem form diagram
# model_star = eqx.combine(diff_model_star, static_model)
# eqstate_star = model_star(structure)
# form_jax_opt = form_from_eqstate(structure, eqstate_star)

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

plotter = Plotter(figsize=(9, 9))

plotter.add(vault_polyline, linestyle="dashed")

for slice in slices:
    plotter.add(slice, draw_as_segment=True, linestyle="solid", color=Color.purple())

# plotter.add(topology)
plotter.add(form_jax, show_reactions=False)

plotter.zoom_extents()
plotter.show()
