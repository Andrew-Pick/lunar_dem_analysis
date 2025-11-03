import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def plot_depth_diameter_vs_latitude(crater_data, bins=20, show_scatter=True, show_binned=True):
    """
    Plots crater depth/diameter ratio as a function of latitude.
    
    Parameters:
    - crater_data (pd.DataFrame): DataFrame with columns 'lat', 'depth_m', 'diameter_m', and optionally 'dD'
    - bins (int): Number of latitude bins for averaging (default: 20)
    - show_scatter (bool): If True, show individual crater points (default: True)
    - show_binned (bool): If True, show binned averages (default: True)
    
    Returns:
    - fig, ax: matplotlib figure and axes objects
    """
    # Calculate depth/diameter ratio if not already present
    if 'dD' not in crater_data.columns:
        crater_data = crater_data.copy()
        crater_data['dD'] = crater_data['depth_m'] / crater_data['diameter_m']
    
    # Remove NaN values
    clean_data = crater_data[['lat', 'dD']].dropna()
    clean_data['lat'] = abs(clean_data['lat'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of individual craters
    if show_scatter:
        ax.scatter(clean_data['lat'], clean_data['dD'], 
                  alpha=0.3, s=10, color='blue', label='Individual craters')
    
    # Binned average
    if show_binned:
        # Create latitude bins
        lat_bins = np.linspace(clean_data['lat'].min(), clean_data['lat'].max(), bins + 1)
        bin_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
        
        # Calculate mean and std for each bin
        bin_means = []
        bin_stds = []
        for i in range(len(lat_bins) - 1):
            mask = (clean_data['lat'] >= lat_bins[i]) & (clean_data['lat'] < lat_bins[i+1])
            bin_data = clean_data[mask]['dD']
            if len(bin_data) > 0:
                bin_means.append(bin_data.mean())
                bin_stds.append(bin_data.std())
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        
        # Plot binned averages with error bars
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, 
                   color='red', linewidth=2, marker='o', markersize=6,
                   capsize=5, label=f'Binned average (n={bins})')
    
    ax.set_xlabel('Latitude (degrees)', fontsize=12)
    ax.set_ylabel('Depth/Diameter Ratio', fontsize=12)
    ax.set_title('Crater Depth/Diameter vs Latitude', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax