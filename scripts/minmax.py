"""
Solve thrust minimization and maximization for an arch.
"""
import os
import yaml

from compas.geometry import allclose

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure
from jax_cem.equilibrium import form_from_eqstate

from mayan_vaults.datastructures import ThrustNetwork
from mayan_vaults.datastructures import create_topology_from_vault

from mayan_vaults import DATA

from mayan_vaults.vaults import create_vault
from mayan_vaults.vaults import Vault
from mayan_vaults.vaults import MayaVault
from mayan_vaults.vaults import CircularVault
from mayan_vaults.vaults import EllipticalVault
from mayan_vaults.vaults import EllipticalTaperedVault

from mayan_vaults.optimization import solve_thrust_minmax_vault

from mayan_vaults.plotting import VaultPlotter
from mayan_vaults.plotting import plot_thrust_minmax_vault

from mayan_vaults.optimization import constraint_thrust_fn
from mayan_vaults.optimization import calculate_start_params


def run_tna_experiment():
    """
    TNA sandbox.
    """
    # Load yaml file with hyperparameters
    with open("minmax.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    vault_type = find_vault_type(config["vault"])
    vault = create_vault(vault_type, **config["vault"])
    print(vault)

    # Analysis
    network = calculate_thrust_network(vault)
    plot_thrust_network(vault, network)

    # Optimization
    networks, _ = solve_thrust_minmax_vault(vault, **config["optimization"])
    plot_thrust_minmax_vault(vault, networks, **config["plotting"])

    # Export results
    export_networks(vault, networks, **config["export"])


def find_vault_type(vault_config: dict) -> type[Vault]:
    """
    Find the vault type from the vault configuration.
    """
    if vault_config["type"] == "maya":
        return MayaVault
    elif vault_config["type"] == "circular":
        return CircularVault
    elif vault_config["type"] == "elliptical":
        return EllipticalVault
    elif vault_config["type"] == "elliptical_tapered":
        return EllipticalTaperedVault
    else:
        raise ValueError(f"Vault type {vault_config['type']} not supported")


def calculate_thrust_network(vault: Vault, check_constraint: bool = False) -> ThrustNetwork:
    """
    Calculate the thrust network for a given vault.
    """
    # Instantiate a topology diagram
    topology = create_topology_from_vault(vault, px0=-1.0)
    print(topology)

    # JAX CEM - form finding
    structure = EquilibriumStructure.from_topology_diagram(topology)
    model = EquilibriumModel.from_topology_diagram(topology)

    # Calculate equilibrium state
    eqstate = model(structure, tmax=1)

    # Generate thrust network
    network = form_from_eqstate(structure, eqstate, ThrustNetwork)

    # Stats
    sw = vault.weight()
    thrust = network.thrust()

    print(f"SW (Vertical load sum): {sw:.2f}")
    print(f"Thrust at support: {thrust:.2f}")
    print(f"Ratio thrust / SW [%]: {100.0 * thrust / sw:.1f}")

    # Constraint check
    if check_constraint:
        params = calculate_start_params(topology)
        constraint_val = constraint_thrust_fn(params, model, structure)
        print(f"{constraint_val=}")

    # Tests
    assert allclose([network.weight()], [network.load_sum_vertical()])
    assert allclose([network.thrust()], [network.load_sum_horizontal()])

    return network
    

def plot_thrust_network(vault, network):
    """
    Plot the thrust network for a given vault.
    """
    plotter = VaultPlotter(figsize=(8, 8))

    plotter.plot_vault(vault, plot_other_half=True)
    
    plotter.plot_vault_blocks(vault)
    plotter.plot_vault_blocks_lines(vault)
    
    plotter.plot_thrust_network(network)
    plotter.zoom_extents()

    plotter.show()


def export_networks(vault: Vault, networks: list[ThrustNetwork], **config: dict):
    """
    Export the thrust networks to a file.
    """
    if config["export_min"] and networks.get("min") is not None:
        network = networks["min"]
        filepath = os.path.join(DATA, f"thrust_min.json")
        network.to_json(filepath)
        print(f"Exported min thrust network to {filepath}")

    if config["export_max"] and networks.get("max") is not None:
        network = networks["max"]        
        filepath = os.path.join(DATA, f"thrust_max.json")
        network.to_json(filepath)
        print(f"Exported max thrust network to {filepath}")

if __name__ == "__main__":
    run_tna_experiment()
