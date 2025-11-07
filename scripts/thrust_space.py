"""
Solve thrust minimization and maximization for an arch.
"""
import os
import yaml
from math import fabs
from fire import Fire

import numpy as np
from compas.colors import Color

from maya_arches.arches import create_arch

from maya_arches import FIGURES

from maya_arches.optimization import solve_thrust_minmax_arch

from maya_arches.plotting import ArchPlotter

from minmax import find_arch_type
from minmax import calculate_thrust_network


def run_tna_space(
    num: int = 10, padding_y: float = 0.5, pad_down: bool = True, pad_up: bool = True, plot_minmax: bool = False):
    """
    TNA sandbox.
    """
    # Load yaml file with hyperparameters
    with open("minmax.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    arch_type = find_arch_type(config["arch"])
    arch = create_arch(arch_type, **config["arch"])
    print(arch)

    # Optimization
    networks_minmax, results = solve_thrust_minmax_arch(arch, **config["optimization"])    
    thrust_min, thrust_max = [results[key]["thrust"] for key in results.keys()]
    ymax, ymin = [results[key]["y"] for key in results.keys()]    

    # Analysis
    p_start = thrust_max   
    p_end = thrust_min
    p_space = np.linspace(p_start, p_end, num=num)
    y_space = np.linspace(ymin, ymax, num=num)    

    # Calculate thrust networks
    networks = []
    for px0, y in zip(p_space, y_space):      
        print(f"px0: {px0}, y: {y}")
        network = calculate_thrust_network(arch, -1.0 * px0, y)
        networks.append(network)

    # Calculate thrust networks padding up
    if pad_up:
        y_diff = ymax - ymin
        y_space_pad = np.linspace(start=ymax, stop=ymax+y_diff, num=num)
        step_size_p = fabs((p_start - p_end) / (num - 1))
        
        for i, y in enumerate(y_space_pad):
            if i == 0:
                continue
            px0 = p_end - (i * step_size_p * 0.75)  # 0.75 is a hot fix
            if px0 < 0.0:
                break
            print(f"px0: {px0}, y: {y}")
            network = calculate_thrust_network(arch, -1.0 * px0, y)
            networks.append(network)

    # # Calculate thrust networks padding down
    if pad_down:
        step_size_y = (ymax - ymin) / num
        num_padding = int(padding_y / step_size_y)
        step_size_p = fabs((p_start - p_end) / (num - 1))    

        y_space_pad = np.linspace(start=ymin-step_size_y, stop=ymin-padding_y, num=num_padding)
        p_space_pad = np.linspace(start=p_start+step_size_p, stop=p_start+(num_padding * step_size_p), num=num_padding)
        for px0, y in zip(p_space_pad, y_space_pad):        
            network = calculate_thrust_network(arch, -1.0 * px0, y)
            networks.append(network)
    
    # Plot
    plotter = ArchPlotter(figsize=(8, 8))
    plotter.plot_arch(arch, plot_other_half=True)
    plotter.zoom_extents()
    
    color = Color.from_rgb255(12, 119, 184)
    color_lightened = color.lightened(75)
    for network in networks:
        if not plot_minmax:
            plotter.plot_thrust_network(network, linewidth=2.0, color=color)    
        else:
            plotter.plot_thrust_network(network, linewidth=1.5, color=color_lightened)

    if plot_minmax:
    #     for network in networks_minmax.values():        
    #         plotter.plot_thrust_network(network, linewidth=3.0, color=color)
        for key, network in networks_minmax.items():
            linestyle = "solid" if key == "max" else "dashed"
            color_blue = Color.from_rgb255(12, 119, 184)
            color_blue_dark = color_blue.darkened(20)
            color = color_blue_dark if key == "max" else color_blue
            draw_as_polyline = key == "min"
            
            plotter.plot_thrust_network(
                network,
                draw_as_polyline=draw_as_polyline,
                linestyle=linestyle,
                linewidth=3.0,
                color=color
                )

    if config["plotting"]["save_plot"]:
        fname = f"thrust_space.pdf"
        if plot_minmax:
            fname = f"thrust_space_minmax.pdf"
        fig_path = os.path.join(FIGURES, fname)
        plotter.save(fig_path, transparent=True, bbox_inches="tight")
        print(f"\nSaved figure to {fig_path}")

    if config["plotting"]["show_plot"]:
        plotter.show()


if __name__ == "__main__":
    Fire(run_tna_space)
