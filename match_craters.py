import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_craters_on_dem(dem_path, body, crater_path, pole, limit, plotname):  # body can be "Moon" or "Merc", pole can be "N" or "S"
    """
    Reads a DEM and plots crater locations on it.
    
    Parameters:
    
    dem_path (str): Path to the DEM file.
    latitudes (array-like): Array of crater latitudes.
    longitudes (array-like): Array of crater longitudes.
    title (str): Title for the plot.
    """

    crater_data = pd.read_csv(crater_path)
    if pole == 'S':
        mask = (crater_data['lat'] < -limit) & (crater_data['diameter_m'] > 8000)
    elif pole == 'N':
        mask = (crater_data['lat'] > limit) & (crater_data['diameter_m'] > 8000)
    latitudes = abs(crater_data['lat'].values[mask])
    longitudes = (crater_data['lon'].values[mask])

    if body == "Moon":
        radius = 1737.4 * 1000  # Moon radius in m
    elif body == "Merc":
        radius = 2439.7 * 1000  # Mercury radius in m

    with rasterio.open(dem_path) as dataset:
        dem_data = dataset.read(1)  # Read the first band
        transform = dataset.transform

        height, width = dem_data.shape
        xres = transform.a  # Pixel width in m
        yres = -transform.e  # Pixel height in m (negative because of the coordinate system)
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + (width * xres)
        ymin = ymax - (height * yres)

        extent = (xmin/1000, xmax/1000, ymin/1000, ymax/1000)  # Convert to km for plotting

        plt.figure(figsize=(10, 8))
        plt.imshow(dem_data, cmap='terrain', extent=extent, origin='upper')
        plt.colorbar(label='Elevation (m)')
        
        # Convert lat/lon to pixel coordinates
        for lat, lon in zip(latitudes, longitudes):
            r = 2 * radius * np.tan(np.pi / 4 - ((lat) * (np.pi / 180)) / 2)
            if pole == 'S':
                theta = (np.pi / 2 - lon * (np.pi / 180)) # Get south pole angle
            elif pole == 'N':
                theta = (3 * np.pi / 2 + lon * (np.pi / 180)) #% (2*np.pi) # Get north pole angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            plt.plot(x / 1000, y / 1000, 'ro')  # Plot crater location in km
            print(r, theta)
            print(x, y)
            #row, col = dataset.index(x, y)
            #plt.plot(col * xres / 1000 + xmin / 1000, row * yres / 1000 + ymax / 1000, 'ro')  # Plot crater location

        plt.title(plotname)
        plt.xlabel('x (km)')
        plt.ylabel('y(km)')
        plt.savefig(f'plots/{plotname}', dpi=300)

plot_craters_on_dem('LDEM_85S_20M.JP2', 'Moon', 'moon_data.csv', 'S', 85, 'test.png')