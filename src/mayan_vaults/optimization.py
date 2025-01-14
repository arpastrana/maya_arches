from math import fabs

from typing import Callable

from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx

from jax import jit
from jax import jacfwd
from jax import value_and_grad

from scipy.optimize import NonlinearConstraint
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize

from compas_cem.diagrams import TopologyDiagram
from compas_cem.diagrams import FormDiagram

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure
from jax_cem.equilibrium import form_from_eqstate

from mayan_vaults.datastructures import ThrustNetwork
from mayan_vaults.datastructures import ThrustNetwork2D


# ------------------------------------------------------------------------------
# Optimization functions
# ------------------------------------------------------------------------------

def rebuild_model_from_params(params: jax.Array, model: EquilibriumModel) -> EquilibriumModel:
    """
    Reconstruct the model from the parameters.
    """
    # unpack parameters
    px = params[0]
    y = params[1]

    # update arrays in place
    node_index = 0
    loads = model.loads.at[node_index, 0:1].set(px)
    xyz = model.xyz.at[node_index, 1:2].set(y)

    # update model pytree with equinox voodoo
    model = eqx.tree_at(
        lambda tree: (tree.loads, tree.xyz),
        model,
        replace=(loads, xyz)
    )

    return model


def minimize_thrust_fn(params: jax.Array, model: EquilibriumModel, structure: EquilibriumStructure) -> float:
    """
    The loss function to minimize thrust at the support.
    """
    # reassemble model
    model = rebuild_model_from_params(params, model)

    # calculate equilibrium state
    eqstate = model(structure, tmax=1)  # TODO: tmax = 100?

    # extract horizontal reaction force vector
    node_index = -1
    reaction_vector = eqstate.reactions[node_index, :1]
    assert reaction_vector.shape == (1,)

    # calculate error
    return jnp.sqrt(jnp.sum(jnp.square(reaction_vector)))


def maximize_thrust_fn(params: jax.Array, model: EquilibriumModel, structure: EquilibriumStructure) -> float:
    """
    The loss function to maximize thrust at the support.
    """
    return -1.0 * minimize_thrust_fn(params, model, structure)


# ------------------------------------------------------------------------------
# Start parameters
# ------------------------------------------------------------------------------

def calculate_start_params(topology: TopologyDiagram) -> jax.Array:
    """
    Extract the starting parameters for the optimization from the topology diagram.
    """
    node_key = 0

    params0 = [
        topology.node_load(node_key)[0],
        topology.node_coordinates(node_key)[1]
    ]
    params0 = jnp.array(params0)
    assert params0.size == 2

    return params0


# ------------------------------------------------------------------------------
# Bounds
# ------------------------------------------------------------------------------

def calculate_params_bounds(vault, tol: float) -> list[tuple[float, float]]:
    """
    Calculate the bounds for the parameters.
    """
    load_bounds = [(None, None)]
    y_bounds = [(vault.height - vault.lintel_height + tol, vault.height - tol)]
    bounds = load_bounds + y_bounds

    return bounds


# ------------------------------------------------------------------------------
# Constraints
# ------------------------------------------------------------------------------

def constraint_fn(params: jax.Array, model: EquilibriumModel, structure: EquilibriumStructure) -> jax.Array:
    """
    The constraint function to ensure the thrust network is within the vault.
    """
    # reassemble model
    model = rebuild_model_from_params(params, model)

    # calculate equilibrium state
    eqstate = model(structure, tmax=1)  # TODO: tmax = 100?

    # extract x coordinates
    x = eqstate.xyz[:, 0]

    return x


def calculate_constraint(
        vault,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        params0: jax.Array) -> NonlinearConstraint:
    """
    Generate the nonlinear constraint object for the optimization.
    """
    partial_constraint_fn = jit(partial(constraint_fn, model=model, structure=structure))

    # Warm start
    _ = partial_constraint_fn(params0)

    # Calculate bounds
    lb = []
    ub = []
    
    # NOTE: This is possible because the nodes in the structure are sorted by JAX CEM
    # Otherwise, we would need to use an index_node mapping to get the block
    for key in structure.nodes:
        block = vault.blocks[key]
        start = block.line_bottom.start
        end = block.line_bottom.end
        lb.append(start.x)
        ub.append(end.x)

    lb = jnp.array(lb)
    ub = jnp.array(ub)

    jac_fn = jit(jacfwd(partial_constraint_fn))

    # Warm start
    _ = jac_fn(params0)

    constraint = NonlinearConstraint(
        fun=partial_constraint_fn,
        lb=lb,
        ub=ub,
        jac=jac_fn,
        keep_feasible=False
    )

    return constraint


# ------------------------------------------------------------------------------
# Solvers
# ------------------------------------------------------------------------------

def solve_thrust_opt(
        vault,
        params0: jax.Array,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        loss_fn: Callable,
        method: str = "SLSQP",
        maxiter: int = 100,
        tol: float = 1e-6,
        tol_bounds: float = 1e-3
) -> OptimizeResult:
    """
    Solve a thrust minmax problem via gradient-based optimization.
    """
    # Calculate initial loss and gradient values
    value_and_grad_fn = jit(value_and_grad(loss_fn))
    _ = value_and_grad_fn(params0, model, structure)

    # Generate box constraints
    bounds = calculate_params_bounds(vault, tol_bounds)

    # Generate inequality constraints
    constraint = calculate_constraint(vault, model, structure, params0)
    constraints = [constraint]

    # Optimize
    result = minimize(
        value_and_grad_fn,
        params0,
        args=(model, structure),        
        method=method,
        jac=True,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        options={"maxiter": maxiter},
        callback=None
    )
    
    return result


def solve_thrust_min(*args, **kwargs) -> FormDiagram:
    """
    Solve a thrust minimization problem via gradient-based optimization.
    """
    return partial(solve_thrust_opt, loss_fn=minimize_thrust_fn)(*args, **kwargs)


def solve_thrust_max(*args, **kwargs) -> FormDiagram:
    """
    Solve a thrust maximization problem via gradient-based optimization.
    """
    return partial(solve_thrust_opt, loss_fn=maximize_thrust_fn)(*args, **kwargs)


# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------


def create_thrust_network_from_opt_result(
        result: OptimizeResult, 
        model: EquilibriumModel, 
        structure: EquilibriumStructure,
        cls: type[ThrustNetwork] = ThrustNetwork2D
        ) -> ThrustNetwork:
    """
    Build a thrust network from the optimization result.
    """
    params_star = result.x
    model_star = rebuild_model_from_params(params_star, model)
    eqstate_star = model_star(structure)

    return form_from_eqstate(structure, eqstate_star, cls)


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def test_thrust_opt_result(network: ThrustNetwork, vault, result: OptimizeResult) -> None:
    """
    Check the optimization result.
    """
    # Check thrust at support
    thrust = network.thrust()

    # Check thrust matches between scipy and form diagram
    msg = f"Thrust at support ({thrust:.2f}) does not match thrust at support from scipy ({fabs(result.fun):.2f})"
    assert jnp.allclose(thrust, fabs(result.fun)), msg

    # Check self weights match
    sw = vault.weight()
    sw_network = network.weight()
    msg = f"The vertical load sum is not equal to the vertical reaction at the support ({sw_network:.2f} != {sw:.2f})"
    assert jnp.allclose(sw_network, sw), msg

    # Check vertical loads match self weight    
    load_sum_vertical = network.load_sum_vertical()
    msg = "The vertical load sum is not equal to the vertical reaction at the support"
    assert jnp.allclose(load_sum_vertical, sw), msg

    # Check that no node but the first has a horizontal load
    keys = [node for node in network.nodes() if node != 0]
    load_sum_x = network.load_sum_horizontal(keys)
    msg = "The network has an undesired horizontal loads"
    assert jnp.allclose(load_sum_x, 0.0), msg
