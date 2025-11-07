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
from compas.geometry import Line
from compas.geometry import Polyline
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.geometry import distance_point_point
from compas.geometry import intersection_segment_plane

from compas.utilities import pairwise
from compas.utilities import remap_values

from compas_plotters import Plotter
from compas_plotters.artists import NetworkArtist

from maya_arches import FIGURES

from maya_arches.arches import Arch
from maya_arches.arches import MayaArch
from maya_arches.arches import CircularArch
from maya_arches.arches import EllipticalArch
from maya_arches.arches import EllipticalTaperedArch

from maya_arches.datastructures import ThrustNetwork


# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

class ArchPlotter(Plotter):
    """
    A wrapper around the compas plotter to plot an arch.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_arch(self, arch: Arch, plot_other_half: bool = True) -> None:
        """
        Plot the arch.
        """
        self.add(
            arch.polyline(),
            linestyle="solid",
            lineweight=2.0,
            draw_points=False
        )

        if plot_other_half:
            plane = Plane((arch.width / 2.0, 0.0, 0.0), (1.0, 0.0, 0.0))
            R = Reflection.from_plane(plane)
            self.add(
                arch.polygon().transformed(R),
                linewidth=0.0,
                facecolor=Color.from_rgb255(240, 240, 240),
                zorder=50
            )

    def plot_arch_blocks(self, arch: Arch) -> None:
        """
        Plot the arch's blocks.
        """
        for i, block in enumerate(arch.blocks.values()):
            # val = i / len(arch.blocks)
            # val = i / (len(arch.blocks) - 1)
            # val = remap_values([val], original_min=0.0, original_max=1.0, target_min=0.2, target_max=1.0)[0]
            # val = 1.0 - val            
            self.add(
                block.polygon(),
                linewidth=0.75,
                edgecolor=Color.black(),
                # linewidth=1.0,
                # edgecolor=Color.grey(),
                # facecolor=Color(val, val, 0.5),
                # facecolor=Color(val, val, val),
                alpha=0.5,
                zorder=50,
            )

    def plot_arch_blocks_lines(self, arch: Arch) -> None:
        """
        Plot the arch's blocks lines.
        """
        for block in arch.blocks.values():
            self.add(
                block.plane_line(),
                draw_as_segment=True,
                # linestyle="dashed",
                # color=Color.from_rgb255(240, 240, 240),
                linestyle=(0, (8, 3)),
                color=Color.black(),
                # linewidth=1.0,
                linewidth=0.5,
                zorder=100
            )

    def plot_thrust_network(
            self,
            network: ThrustNetwork,
            draw_as_polyline: bool = False,
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

        # Draw the thrust network as a polyline
        if draw_as_polyline:
            if not isinstance(linewidth, (int, float)):
                linewidth = linewidth[0]

            self.add(
                Polyline(points.values()),
                draw_points=False, 
                linestyle=linestyle, 
                color=color,
                linewidth=linewidth, 
                zorder=1000
                )
            return

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


    def plot_thrust_network_loads(self, network: ThrustNetwork, scale: float = 1.0) -> None:
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

    def plot_thrust_network_thrusts(self, network: ThrustNetwork, scale: float = 1.0) -> None:
        """
        Plot the thrust network thrusts (horizontal reactions at supports).
        """
        for node in network.nodes():
            if not network.is_node_support(node):
                continue

            thrust_x = [network.node_attribute(node, 'rx'), 0.0, 0.0]
            thrust_y = [0.0, network.node_attribute(node, 'ry'), 0.0]
            thrust_xyz = network.node_attributes(node, ['rx', 'ry', 'rz'])

            for thrust in (thrust_x, thrust_y):
                thrust_scaled = scale_vector(thrust, scale)

                xyz = network.node_coordinates(node)
                xyz = add_vectors(xyz, scale_vector(thrust_scaled, -1.0))

                self.add(
                        Vector(*thrust_scaled),
                        point=Point(*xyz),
                        color=Color.grey()
                    )

    def plot_constraints(self, arch: Arch, network: ThrustNetwork, tol_bounds: float = 1e-3, tol_constraints: float = 1e-3, pointsize: float = 6.0) -> None:
        """
        Plot the constraints.
        """
        color_constraint_lower = Color.from_rgb255(250, 80, 210)
        color_constraint_upper = Color.orange()

        # Special cases
        points_first_last = []
        block_height_avg = sum(block.height() for block in arch.blocks.values()) / len(arch.blocks)

        # First node
        node_key = 0
        point = Point(*network.node_coordinates(node_key))        
        
        # Check lower bound        
        if fabs(point.y - (arch.height - arch.thickness + tol_bounds)) < tol_bounds:
            self.add(
                    point,
                    size=pointsize,
                    facecolor=color_constraint_lower,
                    zorder=2000
                )
            points_first_last.append(point)
        # Check upper bound
        elif fabs((arch.height - tol_bounds) - point.y) < tol_bounds:            
            self.add(
                    point,
                    size=pointsize,
                    facecolor=color_constraint_upper,
                    zorder=2000
                )
            points_first_last.append(point)
            
        # Last node
        node_key = network.number_of_nodes() - 1
        point = Point(*network.node_coordinates(node_key))
        # Check lower bound
        if fabs(point.x) < tol_bounds:
            self.add(
                    point,
                    size=pointsize,
                    facecolor=color_constraint_upper,
                    zorder=2000
                )
            points_first_last.append(point)
        # Check upper bound
        elif fabs(point.x - arch.support_width) <= tol_bounds:
            self.add(
                    point,
                    size=pointsize,
                    facecolor=color_constraint_lower,
                    zorder=2000
                )
            points_first_last.append(point)

        # Intermediate nodes
        for node in network.nodes():

            block = arch.blocks.get(node)
            if block is None:
                continue

            point = Point(*network.node_coordinates(node))

            # Check if the point is close to the special cases
            is_close_to_special = False
            for point_special in points_first_last:                
                if distance_point_point(point, point_special) - tol_constraints <= block_height_avg:
                    is_close_to_special = True
                    break

            if is_close_to_special:
                continue

            # Check constraint plane line
            plane_line = block.plane_line()
            start = plane_line.start
            end = plane_line.end

            if start.y > end.y:
                start, end = end, start
        
            # Check lower bound
            if point.y > -tol_constraints and distance_point_point(point, start) <= tol_constraints:                
                self.add(
                    point,
                    size=pointsize,
                    facecolor=color_constraint_lower,
                    zorder=2000
                )

            # Check upper bound            
            if distance_point_point(point, end) <= tol_constraints:
                print("Is upper bound")
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

def plot_thrust_minmax_arch(
        arch: Arch, 
        networks: dict,
        plot_other_half: bool = True,
        plot_blocks: bool = True,
        plot_blocks_lines: bool = True,
        plot_constraints: bool = True,
        plot_loads: bool = True,
        plot_thrusts: bool = True,
        save_plot: bool = True,
        show_plot: bool = True,
        thrust_linewidth: float = 3.0,
        forcescale: float = 1.0,
        tol_bounds: float = 1e-3,
        tol_constraints: float = 1e-3,
        ) -> None:
    """
    Plot the thrust minimization and maximization results.
    """
    print("\n***** Plotting *****\n")
    plotter = ArchPlotter(figsize=(8, 8))

    plotter.plot_arch(arch, plot_other_half)

    if plot_blocks:
        plotter.plot_arch_blocks(arch)

    if plot_blocks_lines:
        plotter.plot_arch_blocks_lines(arch)

    plotter.zoom_extents()

    for loss_fn_name, network in networks.items():        
        linestyle = "solid" if loss_fn_name == "max" else "dashed"
        color_blue = Color.from_rgb255(12, 119, 184)
        color_blue_dark = color_blue.darkened(20)
        color = color_blue_dark if loss_fn_name == "max" else color_blue
        draw_as_polyline = loss_fn_name == "min"
        
        plotter.plot_thrust_network(
            network,
            draw_as_polyline=draw_as_polyline,
            linestyle=linestyle,
            linewidth=thrust_linewidth,
            color=color
            )

        if plot_constraints:
            plotter.plot_constraints(arch, network, tol_bounds, tol_constraints)

        if plot_loads:
            plotter.plot_thrust_network_loads(network, forcescale)

        if plot_thrusts:
            plotter.plot_thrust_network_thrusts(network, forcescale)

    if save_plot:
        solution_names = "".join(networks.keys())
        if isinstance(arch, MayaArch):
            fig_name = f"{solution_names}_maya_h{int(arch.height)}_w{int(arch.width)}_wh{int(arch.wall_height * 10.0)}_ww{int(arch.wall_width * 10.0)}_lh{int(arch.lintel_height * 10.0)}_n{int(arch.num_blocks)}.pdf"
        elif isinstance(arch, CircularArch):
            fig_name = f"{solution_names}_circle_r{int(arch.radius)}_t{int(arch.thickness * 10)}_n{int(arch.num_blocks)}.pdf"        
        elif isinstance(arch, EllipticalTaperedArch):
            fig_name = f"{solution_names}_ellipse_tapered_h{int(arch.height)}_w{int(arch.width)}_tt{int(arch.thickness_top * 10)}_tb{int(arch.thickness_bottom * 10)}_n{int(arch.num_blocks)}.pdf"
        elif isinstance(arch, EllipticalArch):
            fig_name = f"{solution_names}_ellipse_h{int(arch.height)}_w{int(arch.width)}_t{int(arch.thickness * 10)}_n{int(arch.num_blocks)}.pdf"
        fig_path = os.path.join(FIGURES, fig_name)
        plotter.save(fig_path, transparent=True, bbox_inches="tight")
        print(f"\nSaved figure to {fig_path}")

    if show_plot:
        plotter.show()
    
    print("\nPlotted!")

    plt.close()
