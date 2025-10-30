import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

def plot_dem(dem_path, title):
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
        plt.imshow(dem_data, cmap='terrain', extent=extent, origin='upper')
        plt.colorbar(label='Elevation (m)')
        plt.title(title)
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        plt.show()

    print("Projection:", profile.get("crs"))
    print("Resolution:", profile.get("transform")[0], "meters per pixel")
    print("DEM shape:", dem_data.shape)
