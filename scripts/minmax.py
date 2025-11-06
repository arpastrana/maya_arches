"""
Solve thrust minimization and maximization for an arch.
"""
import os
import yaml

from compas.geometry import allclose

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure
from jax_cem.equilibrium import form_from_eqstate

from maya_arches.datastructures import ThrustNetwork
from maya_arches.datastructures import create_topology_from_arch

from maya_arches import DATA

from maya_arches.arches import create_arch
from maya_arches.arches import Arch
from maya_arches.arches import MayaArch
from maya_arches.arches import CircularArch
from maya_arches.arches import EllipticalArch
from maya_arches.arches import EllipticalTaperedArch

from maya_arches import FIGURES

from maya_arches.optimization import solve_thrust_minmax_arch

from maya_arches.plotting import ArchPlotter
from maya_arches.plotting import plot_thrust_minmax_arch

from maya_arches.optimization import constraint_thrust_fn
from maya_arches.optimization import calculate_start_params


def run_tna_experiment():
    """
    TNA sandbox.
    """
    # Load yaml file with hyperparameters
    with open("minmax.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    arch_type = find_arch_type(config["arch"])
    arch = create_arch(arch_type, **config["arch"])
    print(arch)

    # Analysis
    network = calculate_thrust_network(arch)
    plot_thrust_network(arch, network, config["plotting"])

    # Optimization
    networks, _ = solve_thrust_minmax_arch(arch, **config["optimization"])
    plot_thrust_minmax_arch(arch, networks, **config["plotting"])

    # Export results
    # export_networks(arch, networks, **config["export"])


def find_arch_type(arch_config: dict) -> type[Arch]:
    """
    Find the arch type from the arch configuration.
    """
    if arch_config["type"] == "maya":
        return MayaArch
    elif arch_config["type"] == "circular":
        return CircularArch
    elif arch_config["type"] == "elliptical":
        return EllipticalArch
    elif arch_config["type"] == "elliptical_tapered":
        return EllipticalTaperedArch
    else:
        raise ValueError(f"Arch type {arch_config['type']} not supported")


def calculate_thrust_network(arch: Arch, check_constraint: bool = False) -> ThrustNetwork:
    """
    Calculate the thrust network for a given arch.
    """
    # Instantiate a topology diagram
    topology = create_topology_from_arch(arch, px0=-1.0)
    # topology = create_topology_from_arch(arch, px0=-0.16)
    # topology = create_topology_from_arch(arch, px0=-0.25)
    # topology = create_topology_from_arch(arch, px0=-2.0)
    # topology = create_topology_from_arch(arch, px0=4.0)
    print(topology)

    # JAX CEM - form finding
    structure = EquilibriumStructure.from_topology_diagram(topology)
    model = EquilibriumModel.from_topology_diagram(topology)

    # Calculate equilibrium state
    eqstate = model(structure, tmax=1)

    # Generate thrust network
    network = form_from_eqstate(structure, eqstate, ThrustNetwork)

    # Stats
    sw = arch.weight()
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
    

def plot_thrust_network(arch: Arch, network: ThrustNetwork, config: dict):
    """
    Plot the thrust network for a given arch.
    """
    plotter = ArchPlotter(figsize=(8, 8))

    plotter.plot_arch(arch, plot_other_half=True)    
    plotter.plot_arch_blocks(arch)
    plotter.plot_arch_blocks_lines(arch)
    
    plotter.zoom_extents()

    plotter.plot_thrust_network(network, linewidth=config["thrust_linewidth"])

    # fig_path = os.path.join(FIGURES, f"thrust_network_blocks_horizontal.pdf")
    # plotter.save(fig_path, transparent=True, bbox_inches="tight")
    # print(f"\nSaved figure to {fig_path}")

    plotter.show()


def export_networks(arch: Arch, networks: dict[str, ThrustNetwork], **config: dict):
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
