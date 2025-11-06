from abc import ABC
from abc import abstractmethod

from compas.geometry import Polyline
from compas.geometry import Polygon
from compas.geometry import Point

from maya_arches.blocks import create_blocks


# ------------------------------------------------------------------------------
# Abstract base class
# ------------------------------------------------------------------------------

class Arch(ABC):
    """
    A base class for arches.
    """
    def __init__(self, **kwargs):
        self.blocks = {}

    @property
    @abstractmethod
    def height(self) -> float:
        """
        The height of the arch.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def width(self) -> float:
        """
        The width of the arch.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def thickness(self) -> float:
        """
        The thickness of the arch.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def support_width(self) -> float:
        """
        The width of the arch's support.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def span(self) -> float:
        """
        The span of the arch.
        """
        raise NotImplementedError

    @abstractmethod
    def points(self) -> list[Point]:
        """
        The points on the perimeter of the arch.
        """
        raise NotImplementedError

    @property    
    def span_half(self) -> float:
        """
        Half of the span of the arch.
        """
        return self.span / 2.0
    
    @property
    def num_blocks(self) -> int:
        """
        The number of blocks in the arch.
        """
        return len(self.blocks)

    def blockify(self, num_blocks: int, density: float, slicing_method: int) -> None:
        """
        Generate the arch's blocks.
        """        
        self.blocks = create_blocks(self, num_blocks, density, slicing_method)

    def polyline(self) -> Polyline:
        """
        The silhouette of the arch as a polyline.
        """
        points = self.points()
        return Polyline(points + points[:1])

    def polygon(self) -> Polygon:
        """
        The silhouette of the arch as a polygon.
        """
        return Polygon(self.points())
    
    def weight(self) -> float:
        """
        The weight of the arch.

        Notes
        -----
        The weight of block 0 is excluded from the weight calculation
        because it is a duplicate of block 1.
        """
        return sum(block.weight() for block in self.blocks.values())
    
    def __str__(self, params_other: dict = None) -> str:
        """Return a string representation of the Arch object."""

        params = {
            'height': self.height,
            'width': self.width,            
            'span': self.span,            
            'num_blocks': self.num_blocks
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

def create_arch(
        arch_cls: type[Arch],
        num_blocks: int,
        slicing_method: int,
        block_density: float,
        **arch_kwargs
    ) -> Arch:
    """
    Create a blockified arch.
    """
    arch = arch_cls(**arch_kwargs)    
    arch.blockify(num_blocks, block_density, slicing_method)

    return arch