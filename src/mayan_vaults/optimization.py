"""
TODO: Initialization strategy: minimize loadpath
"""

from warnings import warn

from math import fabs

from typing import Callable
from typing import List

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
from mayan_vaults.datastructures import create_topology_from_vault

from mayan_vaults.vaults import Vault


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
        topology.node_load(node_key)[0],  # horizontal load
        topology.node_coordinates(node_key)[1]  # y coordinate
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
    load_bounds = [(-1000.0, -tol)]  # Load on x always nonpositive
    y_bounds = [(vault.height - vault.thickness + tol, vault.height - tol)]
    bounds = load_bounds + y_bounds

    return bounds


# ------------------------------------------------------------------------------
# Constraints position
# ------------------------------------------------------------------------------

def constraint_position_fn(
        params: jax.Array,
        model: EquilibriumModel,
        structure: EquilibriumStructure) -> jax.Array:
    """
    The constraint function to ensure the thrust network is within the vault.
    """
    # reassemble model
    model = rebuild_model_from_params(params, model)

    # calculate equilibrium state
    eqstate = model(structure, tmax=1)  # TODO: tmax = 100?

    # extract y coordinates of all nodes but the first and last
    x = eqstate.xyz[1:-1, 1]

    return x


def calculate_constraint_position(
        vault,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        params0: jax.Array,
        tol: float = 1e-6) -> NonlinearConstraint:
    """
    Generate the nonlinear constraint object for the optimization.
    """
    partial_constraint_fn = jit(partial(constraint_position_fn, model=model, structure=structure))

    # Warm start
    _ = partial_constraint_fn(params0)

    # Calculate bounds
    lb = []
    ub = []

    # NOTE: This is possible because the nodes in the structure are sorted by JAX CEM
    # Otherwise, we would need to use an index_node mapping to get the block
    for key in structure.nodes:

        # The first node is box constrained
        if key == 0:
            continue

        # The last node (support) is constrained differently
        if key == len(structure.nodes) - 1:
            continue

        block = vault.blocks[key]

        plane_line = block.plane_line()
        start = plane_line.start
        end = plane_line.end

        if start.y > end.y:
            start, end = end, start

        _lb = start.y
        _ub = end.y

        # NOTE: We used to skip the first block on the wall after the corbel to
        # capture a crack at the intersection of the wall's intrados and the ground.
        # But this led to numerical artifacts when calculating the stability domain
        # for mu=0.0. Essentially, we were constraining the thrust to be lower
        # than it should be at this mu. So now we don't skip the first block.
        if start.y <= tol:
            _lb = -vault.height * 50.0

        lb.append(_lb)
        ub.append(_ub)

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
# Constraints thrust
# ------------------------------------------------------------------------------

def constraint_thrust_fn(
        params: jax.Array,
        model: EquilibriumModel,
        structure: EquilibriumStructure) -> jax.Array:
    """
    The constraint function to ensure the thrust fits within the vault.
    """
    # reassemble model
    model = rebuild_model_from_params(params, model)

    # calculate equilibrium state
    eqstate = model(structure, tmax=1)

    # extract xyz coordinates of last node
    node_index = -1
    xyz = eqstate.xyz[node_index, :]

    # extract reaction force on last node
    reaction = eqstate.reactions[node_index, :]

    # project reaction ray onto normal plane on the ground
    normal = jnp.array([0.0, 1.0, 0.0])
    scale = (-xyz @ normal) / (reaction @ normal)
    intersection = xyz + scale * reaction

    # get x component of intersection
    delta = intersection[:1]

    return delta


def calculate_constraint_thrust(
        vault,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        params0: jax.Array) -> NonlinearConstraint:
    """
    Generate the nonlinear constraint object for the optimization.
    """
    partial_constraint_fn = jit(partial(constraint_thrust_fn, model=model, structure=structure))

    # Warm start
    _ = partial_constraint_fn(params0)

    # Calculate bounds
    lb = [0.0]
    ub = [vault.support_width]

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
# Constraints position support
# ------------------------------------------------------------------------------

def constraint_position_support_fn(
        params: jax.Array,
        model: EquilibriumModel,
        structure: EquilibriumStructure) -> jax.Array:
    """
    The constraint function to ensure the support fits within the vault.
    """
    # reassemble model
    model = rebuild_model_from_params(params, model)

    # calculate equilibrium state
    eqstate = model(structure, tmax=1)

    # extract x coordinate of last node
    node_index = -1 
    x = eqstate.xyz[node_index, 0]

    return x


def calculate_constraint_position_support(
        vault,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        params0: jax.Array) -> NonlinearConstraint:
    """
    Generate the nonlinear constraint object for the optimization.
    """
    partial_constraint_fn = jit(partial(constraint_position_support_fn, model=model, structure=structure))

    # Warm start
    _ = partial_constraint_fn(params0)

    # Calculate bounds
    lb = [0.0]
    ub = [vault.support_width]

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
    constraint_position = calculate_constraint_position(vault, model, structure, params0)
    # constraint_thrust = calculate_constraint_thrust(vault, model, structure, params0)
    constraint_position_support = calculate_constraint_position_support(vault, model, structure, params0)

    constraints = [
        constraint_position,
        constraint_position_support
        ]

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

    # Evaluate constraints
    constraints_evaluate_solution(constraints, result)

    return result


def solve_thrust_opt_min(*args, **kwargs) -> FormDiagram:
    """
    Solve a thrust minimization problem via gradient-based optimization.
    """
    return partial(solve_thrust_opt, loss_fn=minimize_thrust_fn)(*args, **kwargs)


def solve_thrust_opt_max(*args, **kwargs) -> FormDiagram:
    """
    Solve a thrust maximization problem via gradient-based optimization.
    """
    return partial(solve_thrust_opt, loss_fn=maximize_thrust_fn)(*args, **kwargs)


# ------------------------------------------------------------------------------
# Solvers on vaults
# ------------------------------------------------------------------------------

def solve_thrust_minmax_vault(
        vault: Vault,
        solve_min: bool = True,
        solve_max: bool = True,
        px0: float = -1.0,
        tol_bounds: float = 1e-3,
        tol: float = 1e-6,
        maxiter: int = 100) -> tuple[dict, dict]:
    """
    Solve the thrust minimization and maximization problems for a given vault geometry.
    """
    # Instantiate a topology diagram
    topology = create_topology_from_vault(vault, px0=px0)

    # JAX CEM - form finding
    structure = EquilibriumStructure.from_topology_diagram(topology)
    model = EquilibriumModel.from_topology_diagram(topology)

    # Calculate start parameters
    # Horizontal load and the 2D coordinates of the origin node
    params0 = calculate_start_params(topology)

    solve_fns = {}
    if solve_min:
        solve_fns["min"] = solve_thrust_opt_min
    if solve_max:
        solve_fns["max"] = solve_thrust_opt_max

    networks = {}
    results = {}

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

        # Generate thrust network
        network = create_thrust_network_from_opt_result(result, model, structure)

        # Check results
        test_thrust_opt_result(network, vault, result)
        networks[solve_fn_name] = network

        # Stats
        sw = vault.weight()
        thrust = network.thrust()

        print(f"SW (Vertical load sum): {sw:.2f}")
        print(f"Thrust at support: {thrust:.2f}")
        print(f"Ratio thrust / SW [%]: {100.0 * thrust / sw:.2f}")

        results[solve_fn_name] = {
            "thrust": thrust,
            "sw": sw
        }

    return networks, results


# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------

def create_thrust_network_from_opt_result(
        result: OptimizeResult,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
        cls: type[ThrustNetwork] = ThrustNetwork
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


def constraints_evaluate_solution(
        constraints: List[NonlinearConstraint],
        result: OptimizeResult,
        tol: float = 1e-6) -> None:
    """
    Evaluates a list of nonlinear optimization constraints at the optimization result.
    """
    print("\nEvaluating constraints at solution")

    for i, constraint in enumerate(constraints):
        cval = constraint.fun(result.x)
        lval = cval - constraint.lb
        uval = constraint.ub - cval
        lval = "Ok" if jnp.abs(jnp.sum(jnp.where(lval <= 0.0, lval, 0.0))) < tol else "Failed"
        uval = "Ok" if jnp.abs(jnp.sum(jnp.where(uval <= 0.0, uval, 0.0))) < tol else "Failed"
        print(f"\tConstraint group {i}\tLower: {lval}\tUpper: {uval}")
    print()
