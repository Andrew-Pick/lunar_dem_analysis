import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from rasterio.transform import Affine

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

def get_crater_profile(dem_data, metadata, crater_lon, crater_lat, crater_diam_km, num_points=100, angle=0):
    """
    Extracts a 2D topographic profile across a crater from a DEM.

    Parameters:
    - dem_data (np.array): The 2D DEM elevation data.
    - metadata (dict): DEM metadata containing the transform.
    - crater_lon, crater_lat (float): Center coordinates of the crater in degrees.
    - crater_diam_km (float): Diameter of the crater in kilometers.
    - num_points (int): Number of points to sample along the profile.
    - angle (float): Angle of the profile in degrees (0=horizontal, 90=vertical).

    Returns:
    - tuple: (distance_km, elevation)
        - distance_km (np.array): Distance along the profile in km.
        - elevation (np.array): Elevation at each point along the profile.
    """
    transform = metadata['transform']
    
    # Convert crater center from lon/lat to pixel coordinates
    # Note: This assumes a simple projection. For polar data, this is an approximation.
    # A more robust solution would use pyproj for coordinate transformations.
    col, row = ~transform * (crater_lon, crater_lat)

    # Define start and end points of the profile line in pixel coordinates
    radius_pixels = (crater_diam_km * 1000 / transform.a) / 2
    angle_rad = np.deg2rad(angle)
    
    start_col = col - radius_pixels * np.cos(angle_rad)
    start_row = row - radius_pixels * np.sin(angle_rad)
    end_col = col + radius_pixels * np.cos(angle_rad)
    end_row = row + radius_pixels * np.sin(angle_rad)

    # Generate sample points along the line
    cols = np.linspace(start_col, end_col, num_points)
    rows = np.linspace(start_row, end_row, num_points)
    
    # Extract elevation values using interpolation
    # map_coordinates expects (row, col) order
    elevation = map_coordinates(dem_data, [rows, cols], order=1, mode='nearest')
    
    # Calculate distance along the profile
    distance_km = np.linspace(-crater_diam_km / 2, crater_diam_km / 2, num_points)
    
    return distance_km, elevation

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
    
    # Convert crater center from lon/lat to pixel coordinates
    col, row = ~transform * (crater_lon, crater_lat)
    
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