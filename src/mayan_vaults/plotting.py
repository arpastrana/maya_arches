import os
import matplotlib.pyplot as plt

from compas.colors import Color

from compas.geometry import Plane
from compas.geometry import Reflection
from compas.geometry import Point
from compas.geometry import Line
from compas.geometry import Polyline
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.geometry import distance_point_point_sqrd

from compas_plotters import Plotter

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
        for block in vault.blocks.values():
            self.add(
                block.line_bottom,
                draw_as_segment=True,
                linestyle="dotted",
                color=Color.grey(),
                lineweight=0.5,
                zorder=100
            )

    def plot_thrust_network(
            self, 
            network,
            linewidth: float = 3.0, 
            linestyle: str = "solid", 
            color: Color = Color.from_rgb255(12, 119, 184),            
            ) -> None:
        """
        Plot the thrust network as a polyline.
        """
        node_keys = list(range(network.number_of_nodes()))
        polyline = Polyline([network.node_coordinates(node) for node in node_keys])

        self.add(
            polyline,
            draw_points=False,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            zorder=1000
        )

    def plot_thrust_network_loads(self, network, scale: float = 1.0) -> None:
        """
        Plot the thrust network loads.
        """
        for node in network.nodes():
            load_x = [network.node_attribute(node, 'qx'), 0.0, 0.0]
            load_y = [0.0, network.node_attribute(node, 'qy'), 0.0]

            for load in (load_x, load_y):
                xyz = network.node_coordinates(node)
                line = Line(xyz, add_vectors(xyz, scale_vector(load, scale)))

                self.add(
                    line,
                    draw_as_segment=True,
                    linestyle="solid",
                    color=Color.from_rgb255(0, 150, 10)
                )

    def plot_thrust_network_thrusts(self, network, scale: float = 1.0) -> None:
        """
        Plot the thrust network thrusts (horizontal reactions at supports).
        """
        for node in network.nodes():
            if not network.is_node_support(node):
                continue

            thrust_x = [network.node_attribute(node, 'rx'), 0.0, 0.0]
            xyz = network.node_coordinates(node)
            line = Line(xyz, add_vectors(xyz, scale_vector(thrust_x, scale)))

            self.add(
                line,
                draw_as_segment=True,
                linestyle="solid",
                color=Color.grey()
                )
            
    def plot_constraints(self, vault, network, tol: float = 1e-3, pointsize: float = 6.0) -> None:
        """
        Plot the constraints.
        """
        color_constraint_extrados = Color.from_rgb255(250, 80, 210)
        color_constraint_intrados = Color.orange()

        for node in network.nodes():

            block = vault.blocks[node]
            point = Point(*network.node_coordinates(node))

            # Check intrados
            for point_intrados in block.points_intrados():
                if distance_point_point_sqrd(point, point_intrados) <= tol:                    
                    self.add(
                        point, 
                        size=pointsize, 
                        facecolor=color_constraint_intrados, 
                        zorder=2000
                    )

            # Check extrados
            for point_extrados in block.points_extrados():
                if distance_point_point_sqrd(point, point_extrados) <= tol:                    
                    self.add(
                        point, 
                        size=pointsize, 
                        facecolor=color_constraint_extrados, 
                        zorder=2000
                    )

# ------------------------------------------------------------------------------
# Experiment code
# ------------------------------------------------------------------------------

def plot_thrust_minmax_vault(
        vault: MayanVault, 
        networks: dict,
        plot_other_half: bool = True,
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
    print("\n***** Plotting *****")
    plotter = VaultPlotter(figsize=(8, 8))

    plotter.plot_vault(vault, plot_other_half)
    plotter.plot_vault_blocks(vault)

    for loss_fn_name, network in networks.items():    
        linestyle = "solid" if loss_fn_name == "max" else "dashed"
        plotter.plot_thrust_network(network, linestyle=linestyle)

        if plot_constraints:
            plotter.plot_constraints(vault, network, tol_bounds)

    plotter.zoom_extents()

    for loss_fn_name, network in networks.items():
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
