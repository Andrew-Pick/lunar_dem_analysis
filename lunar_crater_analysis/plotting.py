import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

def plot_dem(dem_path):
    """
    Reads and plots a Digital Elevation Model (DEM) from the given file path.
    
    Parameters:
    
    dem_path (str): Path to the DEM file.
    title (str): Title for the plot.
    """

    with rasterio.open(dem_path) as dataset:
        dem_data = dataset.read(1)  # Read the first band
        profile = dataset.profile
        transform = dataset.transform

        height, width = dem_data.shape
        xres = transform.a  # Pixel width in m
        yres = -transform.e  # Pixel height in m (negative because of the coordinate system)
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + (width * xres)
        ymin = ymax - (height * yres)

        extent = (xmin/1000, xmax/1000, ymin/1000, ymax/1000)  # Convert to km for plotting

        fig, ax = plt.subplot(figsize=(10, 8))
        im = ax.imshow(dem_data, cmap='terrain', extent=extent, origin='upper')
        fig.colorbar(im, ax=ax, label='Elevation (m)')
        ax.xlabel('x (km)')
        ax.ylabel('y (km)')
        ax.show()

        return fig, ax
