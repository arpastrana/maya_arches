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
from scipy.optimize import minimize

from compas_cem.diagrams import TopologyDiagram
from compas_cem.diagrams import FormDiagram

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure
from jax_cem.equilibrium import form_from_eqstate


# ------------------------------------------------------------------------------
# Optimization functions
# ------------------------------------------------------------------------------

def rebuild_model_from_params(params: jax.Array, model: EquilibriumModel) -> EquilibriumModel:
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


def minimize_thrust_fn(params: jax.Array, model: EquilibriumModel, structure: EquilibriumStructure) -> float:
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


def maximize_thrust_fn(params: jax.Array, model: EquilibriumModel, structure: EquilibriumStructure) -> float:
    """
    The loss function
    """
    return -1.0 * minimize_thrust_fn(params, model, structure)


# ------------------------------------------------------------------------------
# Start parameters
# ------------------------------------------------------------------------------

def calculate_start_params(topology: TopologyDiagram) -> jax.Array:
    """
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


def calculate_constraint(
        vault,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        params0: jax.Array) -> NonlinearConstraint:
    """
    """
    partial_constraint_fn = jit(partial(constraint_fn, model=model, structure=structure))
    # Warm start
    _ = partial_constraint_fn(params0)

    # Calculate bounds
    lb = []
    ub = []

    for block in vault.blocks[::-1]:
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
) -> FormDiagram:
    """
    """
    # print(f"\nSolving for {loss_fn_name} solution")
    print(f"Calculating initial loss and gradient values")
    loss = loss_fn(params0, model, structure)
    value_and_grad_fn = jit(value_and_grad(loss_fn))
    loss, gradient = value_and_grad_fn(params0, model, structure)

    print("Generating box constraints")
    bounds = calculate_params_bounds(vault, tol_bounds)

    print("Generating inequality constraints")
    constraint = calculate_constraint(vault, model, structure, params0)
    constraints = [constraint]

    print("Optimizing")
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

    if not result.success:
        raise ValueError(f"Optimization failed")
    else:
        print(f"Optimization successful")

    # Generate optimized compas cem form diagram
    params_star = result.x
    model_star = rebuild_model_from_params(params_star, model)
    eqstate_star = model_star(structure)
    form_star = form_from_eqstate(structure, eqstate_star)

    node_key_last = len(vault.blocks)
    assert jnp.allclose(fabs(form_star.node_attribute(node_key_last, 'rx')), fabs(result.fun))

    return form_star


def solve_thrust_min(*args, **kwargs) -> FormDiagram:
    """
    """
    return partial(solve_thrust_opt, loss_fn=minimize_thrust_fn)(*args, **kwargs)


def solve_thrust_max(*args, **kwargs) -> FormDiagram:
    """
    """
    return partial(solve_thrust_opt, loss_fn=maximize_thrust_fn)(*args, **kwargs)
