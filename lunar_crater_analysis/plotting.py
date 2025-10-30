import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

def plot_dem(dem_data, profile):
    """
    Reads and plots a Digital Elevation Model (DEM) from the given file path.
    
    Parameters:
    
    dem_path (str): Path to the DEM file.
    title (str): Title for the plot.
    """
    transform = profile['transform']
    height, width = dem_data.shape
    xres = transform.a  # Pixel width in m
    yres = -transform.e  # Pixel height in m (negative because of the coordinate system)
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + (width * xres)
    ymin = ymax - (height * yres)

    extent = (xmin/1000, xmax/1000, ymin/1000, ymax/1000)  # Convert to km for plotting

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dem_data, cmap='terrain', extent=extent, origin='upper')
    fig.colorbar(im, ax=ax, label='Elevation (m)')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    plt.show()

    return fig, ax

def plot_crater_locations(ax, crater_objects, color='red', show_labels=False):
    """
    Plots crater locations on an existing set of axes.

    This function assumes each crater object has a 'center' attribute with
    (x, y) coordinates and an optional 'id' for labeling.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to draw on.
    - crater_objects (list): A list of Crater objects.
    - color (str): The color for the crater markers.
    - show_labels (bool): If True, display the crater ID next to the marker.
    """
    # Important: This assumes your Crater objects will have their global
    # coordinates stored. You might need to adjust your Crater class or
    # keep a separate list of global coordinates.
    for crater in crater_objects:
        # crater.center should be the (x, y) or (col, row) coordinate
        x, y = crater.center
        ax.plot(x, y, 'o', color=color, markersize=5, label='_nolegend_')

        if show_labels:
            ax.text(x + 5, y + 5, crater.id, color=color)

    # Add a single legend entry for all craters
    ax.plot([], [], 'o', color=color, label='Craters')
    ax.legend()