import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from rasterio.transform import Affine
import pyproj
from pyproj import CRS

def calculate_crater_depth(dem_data, crater_center, crater_rim_radius):
    """
    A simple example function to calculate the depth of a crater.

    Parameters:
    - dem_data (np.array): The Digital Elevation Model.
    - crater_center (tuple): The (row, col) of the crater's center.
    - crater_rim_radius (float): The radius of the crater rim in pixels.

    Returns:
    - float: The calculated depth of the crater.
    """
    # This is a simplified example. A real implementation would be more complex.
    
    # Get the elevation at the center of the crater
    center_elevation = dem_data[crater_center]

    # Create a mask for the crater rim
    y, x = np.ogrid[-crater_center[0]:dem_data.shape[0]-crater_center[0], -crater_center[1]:dem_data.shape[1]-crater_center[1]]
    mask = x*x + y*y <= crater_rim_radius*crater_rim_radius

    # For simplicity, let's just take the average elevation of the rim points.
    # A real implementation would need a more robust way to identify the rim.
    rim_elevation = np.mean(dem_data[mask])

    return rim_elevation - center_elevation

def filter_craters(
    crater_data, 
    pole='north', 
    lat_threshold=60, 
    circ_threshold=None, 
    diam_min=None, 
    diam_max=None,
    lat_col='LAT_CIRC_IMG',
    diam_col='DIAM_CIRC_IMG',
    circ_col='circ'
):
    """
    Filter crater data based on location, circularity, and diameter.

    This function is flexible and allows specifying column names for key attributes.

    Parameters:
    - crater_data (pd.DataFrame): DataFrame containing crater data.
    - pole (str): Which pole to filter for: 'north', 'south', or 'both' (default: 'north').
    - lat_threshold (float): Latitude threshold in degrees (default: 60).
    - circ_threshold (float, optional): Minimum circularity ratio. If None, this filter is skipped.
    - diam_min (float, optional): Minimum crater diameter in meters.
    - diam_max (float, optional): Maximum crater diameter in meters.
    - lat_col (str): Name of the latitude column (default: 'lat').
    - diam_col (str): Name of the diameter column (default: 'DIAM_ELLI_MAJOR_IMG').
    - circ_col (str): Name of the circularity column (default: 'circ').

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    # --- Validate required columns ---
    required_cols = [lat_col]
    if circ_threshold is not None:
        required_cols.append(circ_col)
    if diam_min is not None or diam_max is not None:
        required_cols.append(diam_col)
    
    for col in required_cols:
        if col not in crater_data.columns:
            raise ValueError(f"crater_data must contain a '{col}' column for the specified filtering.")

    # Start with a copy of the original data
    filtered = crater_data.copy()

    # --- Latitude Filtering ---
    pole = pole.lower()
    if pole == 'north':
        filtered = filtered[filtered[lat_col] >= lat_threshold]
    elif pole == 'south':
        filtered = filtered[filtered[lat_col] <= -lat_threshold]
    elif pole == 'both':
        filtered = filtered[np.abs(filtered[lat_col]) >= lat_threshold]
    else:
        raise ValueError(f"pole must be 'north', 'south', or 'both', got '{pole}'")
    
    # --- Circularity Filtering ---
    if circ_threshold is not None:
        filtered = filtered[filtered[circ_col] >= circ_threshold]

    # --- Diameter Filtering ---
    if diam_min is not None:
        filtered = filtered[filtered[diam_col] >= diam_min]
    if diam_max is not None:
        filtered = filtered[filtered[diam_col] <= diam_max]
        
    print(f"Filtered to {len(filtered)} craters matching criteria.")
    return filtered


def _transform_coords(from_crs, to_crs, lon, lat):
    """Helper function to transform coordinates between CRS."""
    transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
    proj_lon, proj_lat = transformer.transform(lon, lat)
    return proj_lon, proj_lat


def get_crater_profile(dem_data, metadata, crater_lon, crater_lat, crater_diam_km, angle=0):
    """
    Extracts a 2D topographic profile by sampling actual DEM pixels (no interpolation).
    Uses Bresenham-like line rasterization to get exact pixel values.

    Parameters:
    - dem_data (np.array): The 2D DEM elevation data.
    - metadata (dict): DEM metadata containing the transform.
    - crater_lon, crater_lat (float): Center coordinates of the crater in degrees.
    - crater_diam_km (float): Diameter of the crater in kilometers.
    - angle (float): Angle of the profile in degrees (0=horizontal, 90=vertical).

    Returns:
    - tuple: (distance_km, elevation, pixel_coords)
        - distance_km (np.array): Distance along the profile in km.
        - elevation (np.array): Elevation at each pixel.
        - pixel_coords (np.array): The (row, col) coordinates of each sampled pixel.
    """
    transform = metadata['transform']
    dem_crs = metadata['crs']
    
    # Create a geographic CRS for the Moon using PROJ string
    geographic_crs = CRS.from_proj4("+proj=longlat +a=1737400 +b=1737400 +no_defs")

    # Transform crater center from geographic (lon/lat) to the DEM's projected CRS
    proj_lon, proj_lat = _transform_coords(geographic_crs, dem_crs, crater_lon, crater_lat)

    # Convert projected coordinates to pixel coordinates
    col, row = ~transform * (proj_lon, proj_lat)

    # Define start and end points of the profile line in pixel coordinates
    radius_pixels = (crater_diam_km * 1000 / abs(transform.a)) / 2
    angle_rad = np.deg2rad(angle)
    
    start_col = int(round(col - radius_pixels * np.cos(angle_rad)))
    start_row = int(round(row - radius_pixels * np.sin(angle_rad)))
    end_col = int(round(col + radius_pixels * np.cos(angle_rad)))
    end_row = int(round(row + radius_pixels * np.sin(angle_rad)))

    # Use Bresenham-like algorithm to get all pixels along the line
    pixel_coords = _bresenham_line(start_row, start_col, end_row, end_col)
    
    # Filter out any pixels outside the DEM bounds
    valid_pixels = []
    for r, c in pixel_coords:
        if 0 <= r < dem_data.shape[0] and 0 <= c < dem_data.shape[1]:
            valid_pixels.append((r, c))
    
    pixel_coords = np.array(valid_pixels)
    
    # Extract elevation values at each pixel
    elevation = dem_data[pixel_coords[:, 0], pixel_coords[:, 1]]
    
    # Calculate distance from center for each pixel
    pixel_size_km = abs(transform.a) / 1000.0
    center_idx = len(pixel_coords) // 2
    distances = np.arange(len(pixel_coords)) - center_idx
    distance_km = distances * pixel_size_km
    
    return distance_km, elevation, pixel_coords


def _bresenham_line(y0, x0, y1, x1):
    """
    Bresenham's line algorithm to get all integer pixel coordinates along a line.
    
    Parameters:
    - y0, x0: Starting point (row, col)
    - y1, x1: Ending point (row, col)
    
    Returns:
    - list of (row, col) tuples representing pixels along the line
    """
    pixels = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        pixels.append((y, x))
        
        if x == x1 and y == y1:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return pixels


def detrend_profile(distance, elevation):
    """
    Remove large-scale slope effects from a profile by subtracting a linear least squares fit.
    
    This is useful for analyzing crater morphology without regional slope bias.
    
    Parameters:
    - distance (np.array): Distance along the profile (e.g., in km)
    - elevation (np.array): Elevation values along the profile (e.g., in meters)
    
    Returns:
    - tuple: (detrended_elevation, slope, intercept)
        - detrended_elevation (np.array): Elevation with linear trend removed
        - slope (float): Slope of the best-fit line (elevation units / distance units)
        - intercept (float): Intercept of the best-fit line (elevation units)
    
    Example:
    >>> distance = np.array([0, 1, 2, 3, 4])
    >>> elevation = np.array([100, 102, 104, 106, 108])  # Linear trend
    >>> detrended, slope, intercept = detrend_profile(distance, elevation)
    >>> # detrended will be close to zero (within numerical precision)
    """
    # Perform linear least squares fit
    # np.polyfit with degree 1 gives [slope, intercept]
    coeffs = np.polyfit(distance, elevation, deg=1)
    slope, intercept = coeffs
    
    # Calculate the linear trend
    linear_trend = slope * distance + intercept
    
    # Subtract the trend from the elevation
    detrended_elevation = elevation - linear_trend
    
    return detrended_elevation, slope, intercept


def get_dem_snippet(dem_data, metadata, crater_lon, crater_lat, crater_diam_km, padding_factor=1.5):
    """
    Extracts a square DEM snippet centered on a crater and its corresponding metadata.

    Parameters:
    - dem_data (np.array): The full 2D DEM elevation data.
    - metadata (dict): DEM metadata containing the transform.
    - crater_lon, crater_lat (float): Center coordinates of the crater in degrees.
    - crater_diam_km (float): Diameter of the crater in kilometers.
    - padding_factor (float): Multiplier for the crater diameter to determine the
                              size of the snippet. E.g., 1.5 means the snippet
                              will be 1.5x the crater's diameter.

    Returns:
    - tuple: (dem_snippet, snippet_metadata, center_in_snippet)
        - dem_snippet (np.array): The cropped 2D DEM data.
        - snippet_metadata (dict): The updated metadata for the snippet.
        - center_in_snippet (tuple): The (row, col) of the crater's center
                                     *within the snippet*.
    """
    transform = metadata['transform']
    dem_crs = metadata['crs']

    # Create a geographic CRS for the Moon using PROJ string
    # Moon radius: 1737.4 km, using a sphere for simplicity
    geographic_crs = CRS.from_proj4("+proj=longlat +a=1737400 +b=1737400 +no_defs")

    # Transform crater center from geographic (lon/lat) to the DEM's projected CRS
    proj_lon, proj_lat = _transform_coords(geographic_crs, dem_crs, crater_lon, crater_lat)
    
    # Convert projected coordinates to pixel coordinates
    col, row = ~transform * (proj_lon, proj_lat)
    
    # Calculate snippet size in pixels
    radius_pixels = (crater_diam_km * 1000 / abs(transform.a)) / 2
    snippet_radius_pixels = int(radius_pixels * padding_factor)
    
    # Define the bounding box, ensuring it's within the DEM bounds
    min_row = max(0, int(row - snippet_radius_pixels))
    max_row = min(dem_data.shape[0], int(row + snippet_radius_pixels))
    min_col = max(0, int(col - snippet_radius_pixels))
    max_col = min(dem_data.shape[1], int(col + snippet_radius_pixels))
    
    # Crop the DEM
    dem_snippet = dem_data[min_row:max_row, min_col:max_col]
    
    # Calculate the new transform for the snippet
    # The new origin is the top-left corner of the snippet in the original DEM's coordinates
    new_origin_lon, new_origin_lat = transform * (min_col, min_row)
    
    # Create a copy of the original metadata and update it
    snippet_metadata = metadata.copy()
    snippet_metadata['width'] = dem_snippet.shape[1]
    snippet_metadata['height'] = dem_snippet.shape[0]
    
    # Update the transform
    # The new transform has the same pixel size but a new origin
    new_transform = Affine(transform.a, transform.b, new_origin_lon,
                           transform.d, transform.e, new_origin_lat)
    snippet_metadata['transform'] = new_transform

    # Calculate the crater's center coordinates *relative to the snippet*
    center_in_snippet_row = row - min_row
    center_in_snippet_col = col - min_col
    
    return dem_snippet, snippet_metadata, (center_in_snippet_row, center_in_snippet_col)