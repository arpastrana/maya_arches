import os
from typing import Union
from typing import Tuple

from math import fabs

import matplotlib.pyplot as plt

from compas.colors import Color

from compas.geometry import Plane
from compas.geometry import Reflection
from compas.geometry import Point
from compas.geometry import Vector
from compas.geometry import Polyline
from compas.geometry import Line
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.geometry import distance_point_point_sqrd
from compas.geometry import intersection_segment_plane

from compas.utilities import pairwise

from compas_plotters import Plotter
from compas_plotters.artists import NetworkArtist

from mayan_vaults import FIGURES
from mayan_vaults.vaults import MayanVault


# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

class VaultPlotter(Plotter):
    """
    A wrapper around the compas plotter to plot a vault.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_vault(self, vault, plot_other_half: bool = True) -> None:
        """
        Plot the vault.
        """
        self.add(
            vault.polyline(),
            linestyle="solid",
            lineweight=2.0,
            draw_points=False
        )

        if plot_other_half:
            plane = Plane((vault.width / 2.0, 0.0, 0.0), (1.0, 0.0, 0.0))
            R = Reflection.from_plane(plane)
            self.add(
                vault.polygon().transformed(R),
                linewidth=0.0,
                facecolor=Color.from_rgb255(240, 240, 240),
                zorder=50
            )

    def plot_vault_blocks(self, vault) -> None:
        """
        Plot the vault blocks.
        """
        for i, block in enumerate(vault.blocks.values()):
            val = i / len(vault.blocks)
            self.add(
                block.polygon(),
                linewidth=0.0,
                edgecolor=Color.grey(),
                facecolor=Color(val, val, 0.5),
                alpha=0.5,
                zorder=50,
            )

    def plot_vault_blocks_lines(self, vault) -> None:
        """
        Plot the vault blocks lines.
        """
        for block in vault.blocks.values():
            self.add(
                block.plane_line(),
                draw_as_segment=True,
                linestyle="solid",
                color=Color.from_rgb255(240, 240, 240),  # Color.grey(),
                lineweight=0.1,
                zorder=100
            )

    def plot_thrust_network(
            self,
            network,
            linewidth: Union[float, Tuple[float, float]] = 3.0,
            linestyle: str = "solid",
            color: Color = Color.from_rgb255(12, 119, 184),
            ) -> None:
        """
        Plot the thrust network as a polyline.
        """
        node_keys = list(range(network.number_of_nodes()))
        points = {0: network.node_coordinates(node_keys[0])}

        # Intersect the thrust network with the ground plane
        plane = Plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        for edge in pairwise(node_keys):
            _, v = edge
            segment = [network.node_coordinates(node) for node in edge]

            intersection = intersection_segment_plane(segment, plane)
            if intersection is not None:            
                points[v] = intersection
                break

            _, end = segment
            points[v] = end

        # Calculate line widths
        if isinstance(linewidth, (int, float)):
            edgewidth = {edge: linewidth for edge in network.edges()}
        else:
            edgewidth = {}
            low, high = linewidth
            forces = [fabs(network.edge_force(edge)) for edge in network.edges()]
            forcemin = min(forces)
            forcemax = max(forces)

            for edge in network.edges():
                force = fabs(network.edge_force(edge))
                try:
                    ratio = (force - forcemin) / (forcemax - forcemin)
                except ZeroDivisionError:
                    ratio = 1.0
                width = (1.0 - ratio) * low + ratio * high
                edgewidth[edge] = width

        # Draw the thrusts edges
        for edge in network.edges():
            u, v = edge
            start = points.get(u)
            end = points.get(v)

            if start is None or end is None:
                continue

            self.add(
                Line(start, end),
                draw_points=False,
                draw_as_segment=True,
                linestyle=linestyle,
                color=color,
                linewidth=edgewidth[edge],
                zorder=1000
            )

    def plot_thrust_network_loads(self, network, scale: float = 1.0) -> None:
        """
        Plot the thrust network loads.
        """
        color_green = Color.from_rgb255(0, 150, 10)
        color_gray = Color.grey()

        for node in network.nodes():
            load_x = [network.node_attribute(node, 'qx'), 0.0, 0.0]
            load_y = [0.0, network.node_attribute(node, 'qy'), 0.0]

            loads_colors = [(load_x, load_y), (color_gray, color_green)]
            for load, color in zip(*loads_colors):
                load_scaled = scale_vector(load, scale)

                xyz = network.node_coordinates(node)
                xyz = add_vectors(xyz, scale_vector(load_scaled, -1.0))

                self.add(
                    Vector(*load_scaled),
                    point=Point(*xyz),
                    color=color
                )

    def plot_thrust_network_thrusts(self, network, scale: float = 1.0) -> None:
        """
        Plot the thrust network thrusts (horizontal reactions at supports).
        """
        for node in network.nodes():
            if not network.is_node_support(node):
                continue

            thrust_x = [network.node_attribute(node, 'rx'), 0.0, 0.0]
            thrust_xyz = network.node_attributes(node, ['rx', 'ry', 'rz'])

            for thrust in (thrust_xyz, ):
                thrust_scaled = scale_vector(thrust, scale)

                xyz = network.node_coordinates(node)
                xyz = add_vectors(xyz, scale_vector(thrust_scaled, -1.0))

                self.add(
                        Vector(*thrust_scaled),
                        point=Point(*xyz),
                        color=Color.grey()
                    )

    def plot_constraints(self, vault, network, tol: float = 1e-3, pointsize: float = 6.0) -> None:
        """
        Plot the constraints.
        """
        color_constraint_lower = Color.from_rgb255(250, 80, 210)
        color_constraint_upper = Color.orange()

        # First node
        node_key = 0
        point = Point(*network.node_coordinates(node_key))       
        # Check lower bound
        if point.y <= (vault.height - vault.lintel_height + tol)
            self.add(
                    point,
                    size=pointsize,
                    facecolor=color_constraint_lower,
                    zorder=2000
                )
        # Check upper bound
        elif point.y >= (vault.height - tol):
            self.add(
                    point,
                    size=pointsize,
                    facecolor=color_constraint_upper,
                    zorder=2000
                )

        # Intermediate nodes
        for node in network.nodes():

            block = vault.blocks.get(node)
            if block is None:
                continue

            point = Point(*network.node_coordinates(node))

            # Check constraint plane line
            plane_line = block.plane_line()
            start = plane_line.start
            end = plane_line.end

            if start.y > end.y:
                start, end = end, start

            # Check lower bound            
            if point.y > 0.0 and distance_point_point_sqrd(point, start) <= tol:
                self.add(
                    point,
                    size=pointsize,
                    facecolor=color_constraint_lower,
                    zorder=2000
                )

            # Check upper bound            
            if distance_point_point_sqrd(point, end) <= tol:
                self.add(
                    point,
                    size=pointsize,
                    facecolor=color_constraint_upper,
                    zorder=2000
                )

# ------------------------------------------------------------------------------
# Artists
# ------------------------------------------------------------------------------

class ThrustNetworkArtist(NetworkArtist):
    """
    A wrapper around the compas network artist to plot a thrust network.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# ------------------------------------------------------------------------------
# Experiment code
# ------------------------------------------------------------------------------

def plot_thrust_minmax_vault(
        vault: MayanVault, 
        networks: dict,
        plot_other_half: bool = True,
        plot_blocks: bool = True,
        plot_blocks_lines: bool = True,
        plot_constraints: bool = True,
        plot_loads: bool = True,
        plot_thrusts: bool = True,
        save_plot: bool = True,
        show_plot: bool = True,
        forcescale: float = 1.0,
        tol_bounds: float = 1e-3,
        ) -> None:
    """
    Plot the thrust minimization and maximization results.
    """
    print("\n***** Plotting *****\n")
    plotter = VaultPlotter(figsize=(8, 8))

    plotter.plot_vault(vault, plot_other_half)

    if plot_blocks:
        plotter.plot_vault_blocks(vault)

    if plot_blocks_lines:
        plotter.plot_vault_blocks_lines(vault)

    plotter.zoom_extents()

    for loss_fn_name, network in networks.items():
        linestyle = "solid" if loss_fn_name == "max" else "solid"

        plotter.plot_thrust_network(network, linestyle=linestyle, linewidth=(2, 7))

        if plot_constraints:
            plotter.plot_constraints(vault, network, tol_bounds)

        if plot_loads:
            plotter.plot_thrust_network_loads(network, forcescale)

        if plot_thrusts:
            plotter.plot_thrust_network_thrusts(network, forcescale)

    if save_plot:
        fig_name = f"minmax_h{int(vault.height)}_w{int(vault.width)}_wh{int(vault.wall_height)}_ww{int(vault.wall_width)}_lh{int(vault.lintel_height)}.pdf"
        fig_path = os.path.join(FIGURES, fig_name)
        plotter.save(fig_path, transparent=True, bbox_inches="tight")
        print(f"\nSaved figure to {fig_path}")

    if show_plot:
        plotter.show()

    plt.close()
