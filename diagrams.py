from compas_cem.diagrams import TopologyDiagram
from compas_cem.elements import Node
from compas_cem.elements import TrailEdge
from compas_cem.loads import NodeLoad
from compas_cem.supports import NodeSupport

from compas.utilities import pairwise


# ------------------------------------------------------------------------------
# Instantiate a topology diagram
# ------------------------------------------------------------------------------

def create_topology_from_vault(vault, block_density: float, px0: float) -> TopologyDiagram:
    """
    Create a topology diagram with nodes, loads, trail edges and supports from a vault.
    """
    topology = TopologyDiagram()

    node_keys = add_nodes(topology, vault)

    add_loads(topology, node_keys, vault, block_density, px0)
    add_trail_edges(topology, node_keys, vault)
    add_supports(topology, node_keys)
    
    topology.build_trails(auxiliary_trails=False)
    
    return topology

# ------------------------------------------------------------------------------
# Add Nodes
# ------------------------------------------------------------------------------

def add_nodes(topology: TopologyDiagram, vault) -> list[int]:
    """
    """
    num_blocks = len(vault.blocks)
    node_keys = []
    for i in range(num_blocks + 1):
        factor = 1.0 - i / (num_blocks)
        point = [factor * vault.width * 0.5, vault.height, 0.0]
        key = topology.add_node(Node(i, point))
        node_keys.append(key)

    return node_keys

# ------------------------------------------------------------------------------
# Set Supports Nodes
# ------------------------------------------------------------------------------

def add_supports(topology: TopologyDiagram, node_keys: list[int]) -> None:
    """
    """
    node_key_last = node_keys[-1]
    topology.add_support(NodeSupport(node_key_last))

# ------------------------------------------------------------------------------
# Add Loads
# ------------------------------------------------------------------------------

def add_loads(topology: TopologyDiagram, node_keys: list[int], vault, block_density: float, px0: float) -> None:
    """
    Add loads to the topology diagram.
    """
    assert (len(node_keys) - 1) == len(vault.blocks), f"nodes: {len(node_keys) - 1} vs. {len(vault.blocks)}"

    node_keys_no_first = node_keys[1:]
    node_keys_no_first_reversed = node_keys_no_first[::-1]
    node_key_first = node_keys[0]
    node_key_second = node_keys[1]

    # Apply loads to all nodes except the first
    for key, block in zip(node_keys_no_first_reversed, vault.blocks):
        py = block.weight(block_density * -1.0)
        load = [0.0, py, 0.0]
        topology.add_load(NodeLoad(key, load))

    # Modify the first and second node loads
    block_last = vault.blocks[-1]  # The last block is the lintel roof
    py0 = block_last.weight(block_density * -1.0) / 2.0

    topology.add_load(NodeLoad(node_key_first, [px0, py0, 0.0]))
    topology.add_load(NodeLoad(node_key_second, [0.0, py0, 0.0]))

# ------------------------------------------------------------------------------
# Add Trail Edges
# ------------------------------------------------------------------------------

def add_trail_edges(topology: TopologyDiagram, node_keys: list[int], vault) -> None:
    """
    """
    node_keys_reversed = node_keys[::-1]
    for (u, v), block in zip(pairwise(node_keys_reversed), vault.blocks):
        topology.add_edge(TrailEdge(u, v, length=-1.0, plane=block.plane))
