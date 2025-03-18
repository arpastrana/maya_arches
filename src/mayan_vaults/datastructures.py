from math import fabs

from compas.geometry import add_vectors
from compas.geometry import Plane
from compas.geometry import Point
from compas.geometry import Vector
from compas.utilities import pairwise

from compas_cem.diagrams import TopologyDiagram
from compas_cem.diagrams import FormDiagram
from compas_cem.elements import Node
from compas_cem.elements import TrailEdge
from compas_cem.loads import NodeLoad
from compas_cem.supports import NodeSupport


# ------------------------------------------------------------------------------
# Instantiate a topology diagram
# ------------------------------------------------------------------------------

def create_topology_from_vault(vault, px0: float = -1.0) -> TopologyDiagram:
    """
    Create a topology diagram with nodes, loads, trail edges and supports from a vault.
    """
    topology = TopologyDiagram()

    node_keys = add_nodes(topology, vault)
    add_loads(topology, vault, node_keys, px0)
    add_trail_edges(topology, node_keys, vault)
    add_supports(topology, node_keys)
    
    topology.build_trails(auxiliary_trails=False)
    
    return topology

# ------------------------------------------------------------------------------
# Add Nodes
# ------------------------------------------------------------------------------

def add_nodes(topology: TopologyDiagram, vault) -> list[int]:
    """
    Add nodes to the topology diagram.

    Notes
    -----
    The number of nodes is equal to the number of blocks plus two.
    The first and the last nodes are special: 
    they represent the origin and the support of a trail, respectively.
    """
    num_nodes = len(vault.blocks) + 2

    node_keys = []
    for i in range(num_nodes):
        factor = 1.0 - i / (num_nodes - 1)
        point = [
            factor * vault.width * 0.5, 
            vault.height - vault.lintel_height * 0.5, 
            0.0
            ]
        key = topology.add_node(Node(i, point))
        node_keys.append(key)

    assert (len(node_keys)) == len(vault.blocks) + 2, f"Nodes: {len(node_keys)} vs. {len(vault.blocks) + 2}"

    return node_keys

# ------------------------------------------------------------------------------
# Set Supports Nodes
# ------------------------------------------------------------------------------

def add_supports(topology: TopologyDiagram, node_keys: list[int]) -> None:
    """
    Add supports.
    """
    node_key_last = node_keys[-1]
    topology.add_support(NodeSupport(node_key_last))

# ------------------------------------------------------------------------------
# Add Loads
# ------------------------------------------------------------------------------

def add_loads(topology: TopologyDiagram, vault, node_keys: list[int], px0: float) -> None:
    """
    Add loads to the topology diagram.
    """
    # node_keys_lintel = node_keys[:1]

    # Apply vertical loads to all nodes
    for key in topology.nodes():
        
        # skip nodes without a block (first and last nodes)
        block = vault.blocks.get(key)
        if block is None:
            continue

        py = block.weight() * -1.0

        # Apply only a portion of the load to the lintel nodes
        # if key in node_keys_lintel:
        #     py = py / len(node_keys_lintel)

        load = [0.0, py, 0.0]
        topology.add_load(NodeLoad(key, load))

    # Add horizontal load to the first node    
    node_key_first = node_keys[0]
    load_vector_first = add_vectors([px0, 0.0, 0.0], topology.node_load(node_key_first))
    load_first = NodeLoad(node_key_first, load_vector_first)
    topology.add_load(load_first)


# ------------------------------------------------------------------------------
# Add Trail Edges
# ------------------------------------------------------------------------------

def add_trail_edges(topology: TopologyDiagram, node_keys: list[int], vault) -> None:
    """
    """
    for u, v in pairwise(node_keys):
        block = vault.blocks.get(v)
        
        if block is None:            
            plane = Plane(Point(0.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0))
        else:
            plane = block.plane()

        topology.add_edge(TrailEdge(u, v, length=-1.0, plane=plane))


# ------------------------------------------------------------------------------
# Thrust Network
# ------------------------------------------------------------------------------


class ThrustNetwork(FormDiagram):
    pass


class ThrustNetwork2D(ThrustNetwork):
    """
    A thrust network is a CEM form diagram with a few extra bells and whistles.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def weight(self) -> float:
        """
        The weight of the thrust network is the sum of the vertical reactions at the supports.
        """
        return sum(self.node_attribute(key, 'ry') for key in self.support_nodes())
    
    def thrust(self) -> float:
        """
        The thrust at the support is the sum of the horizontal reactions at the support nodes.
        """
        return sum(fabs(self.node_attribute(node, 'rx')) for node in self.support_nodes())

    def load_sum_vertical(self, keys: list[int] = None) -> list[float]:
        """
        The vertical load sum is the sum of the vertical loads at all nodes.
        """
        if not keys:
            keys = self.nodes()
        return sum(fabs(self.node_attribute(node, 'qy')) for node in keys)
    
    def load_sum_horizontal(self, keys: list[int] = None) -> list[float]:
        """
        The horizontal load sum is the sum of the horizontal loads at all nodes.
        """
        if not keys:
            keys = self.nodes()
        return sum(fabs(self.node_attribute(node, 'qx')) for node in keys)
