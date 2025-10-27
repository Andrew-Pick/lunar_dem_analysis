import rasterio
import matplotlib.pyplot as plt
import numpy as np

dataset = rasterio.open('LDEM_85S_40M.JP2')

print(dataset.transform)
print(dataset.crs)
print(dataset.transform * (0, 0))  # Top-left corner
print(dataset.transform * (dataset.width, dataset.height))  # Bottom-right corner

band1 = dataset.read(1)
print(band1.shape)
print(band1[dataset.height // 2, dataset.width // 2])  # Center pixel value
print(band1)

x, y = 2000, -1500
row, col = dataset.index(x, y)
print(f"Coordinates ({x}, {y}) correspond to row {row}, column {col}")