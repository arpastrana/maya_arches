"""
Solve thrust minimization and maximization for a Mayan vault in 2D.
"""
import yaml

from mayan_vaults.vaults import create_vault
from mayan_vaults.optimization import solve_thrust_minmax_vault
from mayan_vaults.plotting import plot_thrust_minmax_vault


def run_minmax_experiment():
    """
    Run a thrust minimization and maximization experiment.
    """
    # load yaml file with hyperparameters
    with open(f"minmax.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    vault = create_vault(**config["vault"])
    networks, _ = solve_thrust_minmax_vault(vault, **config["optimization"])
    plot_thrust_minmax_vault(vault, networks, **config["plotting"])


if __name__ == "__main__":
    run_minmax_experiment()
