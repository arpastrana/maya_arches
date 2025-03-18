"""
Solve thrust minimization and maximization for a Mayan vault in 2D.
"""
import yaml

from compas.geometry import allclose

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure
from jax_cem.equilibrium import form_from_eqstate

from mayan_vaults.datastructures import ThrustNetwork2D
from mayan_vaults.datastructures import create_topology_from_vault

from mayan_vaults.vaults import create_vault

from mayan_vaults.optimization import solve_thrust_minmax_vault

from mayan_vaults.plotting import VaultPlotter
from mayan_vaults.plotting import plot_thrust_minmax_vault


def run_tna_experiment():
    """
    TNA sandbox.
    """
    # load yaml file with hyperparameters
    with open(f"tna.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    vault = create_vault(**config["vault"])
    print(vault)

    # Instantiate a topology diagram
    topology = create_topology_from_vault(vault, px0=-3.7)
    print(topology)

    # JAX CEM - form finding
    structure = EquilibriumStructure.from_topology_diagram(topology)
    model = EquilibriumModel.from_topology_diagram(topology)

    # Calculate equilibrium state
    eqstate = model(structure, tmax=1)

    # Generate thrust network
    network = form_from_eqstate(structure, eqstate, ThrustNetwork2D)

    # Stats
    sw = vault.weight()
    thrust = network.thrust()

    print(f"SW (Vertical load sum): {sw:.2f}")
    print(f"Thrust at support: {thrust:.2f}")
    print(f"Ratio thrust / SW [%]: {100.0 * thrust / sw:.1f}")

    print(f"\nLoads:")
    for node in network.nodes():
        print(f"\tNode {node}: {network.node_attribute(node, 'qy')}")

    print(f"\nXYZ:")
    for node in network.nodes():
        print(f"\tNode {node}: {network.node_coordinates(node)}")

    # Tests
    assert allclose([network.weight()], [network.load_sum_vertical()])
    assert allclose([network.thrust()], [network.load_sum_horizontal()])
    
    # Plotting
    plotter = VaultPlotter(figsize=(8, 8))
    
    plotter.plot_vault(vault, plot_other_half=True)
    plotter.plot_vault_blocks(vault)

    # plotter.plot_thrust_network(network)
    plotter.add(network, show_nodes=True, nodesize=0.5, show_reactions=False, show_loads=False)
    
    plotter.zoom_extents()
    plotter.show()

    # Optimization
    networks, _ = solve_thrust_minmax_vault(vault, **config["optimization"])

    for network in networks.values():
        print(f"\nLoads:")
        for node in network.nodes():
            print(f"\tNode {node}: qx={network.node_attribute(node, 'qx')}, qy={network.node_attribute(node, 'qy')}")

        print(f"\nXYZ:")
        for node in network.nodes():
            print(f"\tNode {node}: {network.node_coordinates(node)}")

    plot_thrust_minmax_vault(vault, networks, **config["plotting"])


if __name__ == "__main__":
    run_tna_experiment()
