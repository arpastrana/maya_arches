from compas.colors import Color

from compas.geometry import Plane
from compas.geometry import Reflection
from compas.geometry import Point
from compas.geometry import Line
from compas.geometry import Polyline
from compas.geometry import add_vectors
from compas.geometry import scale_vector

from compas_plotters import Plotter

# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

def plot_vault(
        vault, 
        forms_star: dict,
        plot_thrusts: bool,
        plot_constraints: bool,
        plot_loads: bool,
        plot_other_half: bool,
        forcescale: float,
        ):
    """
    """
    print("\nPlotting")
    plotter = Plotter(figsize=(8, 8))

    # Plot vault
    plotter.add(vault.polyline(), linestyle="solid", lineweight=2.0, draw_points=False)

    # Plot blocks
    for block in vault.blocks:
        plotter.add(
            block.line_bottom,
            draw_as_segment=True,
            linestyle="dotted",
            color=Color.grey(),
            lineweight=0.5,
            zorder=100
        )

    # Plot other half
    if plot_other_half:
        plane = Plane((vault.width / 2.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        R = Reflection.from_plane(plane)
        plotter.add(
            vault.polygon().transformed(R),
            linewidth=0.0,
            facecolor=Color.from_rgb255(240, 240, 240),
            zorder=50
            )

    # Plot forms
    color_blue = Color.from_rgb255(12, 119, 184)
    for loss_fn_name, form_star in forms_star.items():
    
        linestyle = "solid"
        if loss_fn_name == "min":
            linestyle = "dashed"

        # Plot form as polyline
        _polyline = Polyline([form_star.node_coordinates(node) for node in node_keys])      
        plotter.add(
            _polyline,
            draw_points=False,
            linestyle=linestyle,
            color=color_blue,
            linewidth=3.0,
            zorder=1000
        )

        # Plot constraints
        if plot_constraints:
        
            color_constraint_extrados = Color.from_rgb255(250, 80, 210)
            color_constraint_intrados = Color.orange()
            constraint_plot_tol = 1e-6

            for node, block in zip(node_keys_no_first[::-1], vault.blocks):
                slice = block.line_bottom
                x, y, z = form_star.node_coordinates(node)

                # Check extrados
                x_constraint = slice.start.x
                if x - constraint_plot_tol <= x_constraint:                            
                    point = Point(x, y, z)
                    plotter.add(point, size=6.0, facecolor=color_constraint_extrados, zorder=2000)

                # Check intrados
                x_constraint = slice.end.x
                if x + constraint_plot_tol >= x_constraint:
                    point = Point(x, y, z)
                    plotter.add(point, size=6.0, facecolor=color_constraint_intrados, zorder=2000)

            x, y, z = form_star.node_coordinates(node_key_first)
            if y - constraint_plot_tol <= vault.height - vault.lintel_height + tol_bounds:
                point = Point(x, y, z)
                plotter.add(point, size=6.0, facecolor=color_constraint_intrados, zorder=2000)

            if y + constraint_plot_tol >= vault.height - tol_bounds:
                point = Point(x, y, z)
                plotter.add(point, size=6.0, facecolor=color_constraint_extrados, zorder=2000)

        # Plot loads
        if plot_loads:
            for node in form_star.nodes():
                load_y = [0.0, form_star.node_attribute(node, 'qy'), 0.0]
                xyz = form_star.node_coordinates(node)
                line = Line(xyz, add_vectors(xyz, scale_vector(load_y, forcescale)))
                plotter.add(
                    line,
                    draw_as_segment=True,
                    linestyle="solid",
                    color=Color.from_rgb255(0, 150, 10)
                )

    plotter.zoom_extents()

    if plot_thrusts:
        for loss_fn_name, form_star in forms_star.items():
            # _nodes = [node_key_first, node_key_last, node_key_last]
            # load = [form_star.node_attribute(node_key_first, 'qx'), 0.0, 0.0]
            # thrust_x = [form_star.node_attribute(node_key_last, 'rx'), 0.0, 0.0]
            # thrust_y = [0.0, form_star.node_attribute(node_key_last, 'ry'), 0.0]
            # _forces = [load, thrust_x, thrust_y]

            _nodes = [node_key_first, node_key_last]
            load = [form_star.node_attribute(node_key_first, 'qx'), 0.0, 0.0]
            thrust_x = [form_star.node_attribute(node_key_last, 'rx'), 0.0, 0.0]        
            _forces = [load, thrust_x]

            print()
            for node, force in zip(_nodes, _forces):
                print(f"Thrust: {force}")
                xyz = form_star.node_coordinates(node)
                line = Line(xyz, add_vectors(xyz, scale_vector(force, -1.0 * forcescale)))
                plotter.add(
                    line,
                    draw_as_segment=True,
                    linestyle="solid",
                    linewidth=2.0,
                    color=Color.grey()
                )

    if save_plot:
        plotter.save(f"figures/minmax.pdf", transparent=True, bbox_inches="tight")

    plotter.show()
