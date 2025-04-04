from abc import ABC
from abc import abstractmethod

from compas.geometry import Polyline
from compas.geometry import Polygon
from compas.geometry import Point


# ------------------------------------------------------------------------------
# Abstract base class
# ------------------------------------------------------------------------------

class Vault(ABC):
    """
    A base class for vaults.
    """
    def __init__(self):
        self.blocks = {}

    @property
    @abstractmethod
    def height(self) -> float:
        """
        The height of the vault.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def width(self) -> float:
        """
        The width of the vault.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def thickness(self) -> float:
        """
        The thickness of the vault.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def span(self) -> float:
        """
        The span of the vault.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def span_half(self) -> float:
        """
        Half of the span of the vault.
        """
        raise NotImplementedError

    @abstractmethod
    def points(self) -> list[Point]:
        """
        The points on the perimeter of the vault.
        """
        raise NotImplementedError

    @abstractmethod
    def blockify(self, num_blocks: int, density: float, slicing_method: int) -> None:
        """
        Create blocks from the vault.
        """
        raise NotImplementedError

    def polyline(self):
        """
        The silhouette of the vault as a polyline.
        """
        points = self.points()
        return Polyline(points + points[:1])

    def polygon(self):
        """
        The silhouette of the vault as a polygon.
        """
        return Polygon(self.points())
    
    def weight(self) -> float:
        """
        The weight of the vault.

        Notes
        -----
        The weight of block 0 is excluded from the weight calculation
        because it is a duplicate of block 1.
        """
        return sum(block.weight() for block in self.blocks.values())
    
    def __str__(self, params_other: dict = None) -> str:
        """Return a string representation of the Vault object."""

        params = {
            'height': self.height,
            'width': self.width,            
            'span': self.span,            
            'num_blocks': len(self.blocks)
        }

        if isinstance(params_other, dict):
            params.update(params_other)

        params_formatted = []
        for key, value in params.items():
            if isinstance(value, float):
                value = f"{value:.2f}"
            params_formatted.append(f"\t{key}={value}")

        return f"{self.__class__.__name__}(\n" + "\n".join(params_formatted) + "\n)"


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def create_vault(
        vault_cls: type[Vault],
        num_blocks: int,
        slicing_method: int,
        block_density: float,
        **vault_kwargs
    ) -> Vault:
    """
    Create a blockified vault.
    """
    vault = vault_cls(**vault_kwargs)    
    vault.blockify(num_blocks, block_density, slicing_method)

    return vault