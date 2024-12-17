from vaults import HalfMayanVault2D
from slicing import slice_vault

from compas_plotters import Plotter
from compas.colors import Color


height = 10.0
width = 8.0
wall_width = 2.0
wall_height = 5.0
lintel_height = 2.0


vault = HalfMayanVault2D(
    height,
    width,
    wall_height,
    wall_width,
    lintel_height
)

vault_polyline = vault.polyline()
vault_polygon = vault.polygon()

slices = slice_vault(vault, num_slices=13)
for i, slice in enumerate(slices):
    print(f"Slice {i}:\tLength:{slice.length:.2f}")
    assert slice.length <= vault.width / 2.0
plotter = Plotter(figsize=(9, 9))
plotter.add(vault_polyline, linestyle="dashed")

for slice in slices:
    plotter.add(slice, draw_as_segment=True, linestyle="solid", color=Color.purple())

plotter.zoom_extents()
plotter.show()
