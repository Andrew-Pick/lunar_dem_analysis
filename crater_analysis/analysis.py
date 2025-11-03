import numpy as np
import pandas as pd

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

def filter_craters(crater_data, pole='north', lat_threshold=60):
    """
    Filter crater data to include only craters from a specific polar region.
    
    Parameters:
    - crater_data (pd.DataFrame): DataFrame containing crater data with a 'lat' column
    - pole (str): Which pole to filter for: 'north', 'south', or 'both' (default: 'north')
    - lat_threshold (float): Latitude threshold in degrees (default: 60)
                            For north pole: craters with lat >= lat_threshold
                            For south pole: craters with lat <= -lat_threshold
    
    Returns:
    - pd.DataFrame: Filtered DataFrame containing only polar craters
    
    Examples:
    >>> # Get only North polar craters (lat >= 60°)
    >>> north_craters = filter_craters_by_latitude(crater_data, pole='north', lat_threshold=60)
    
    >>> # Get only South polar craters (lat <= -80°)
    >>> south_craters = filter_craters_by_latitude(crater_data, pole='south', lat_threshold=80)
    
    >>> # Get both polar regions (|lat| >= 70°)
    >>> polar_craters = filter_craters_by_latitude(crater_data, pole='both', lat_threshold=70)
    """
    if 'lat' not in crater_data.columns:
        raise ValueError("crater_data must contain a 'lat' column")
    
    pole = pole.lower()
    
    if pole == 'north':
        filtered = crater_data[crater_data['lat'] >= lat_threshold].copy()
        print(f"Filtered to {len(filtered)} North polar craters (lat >= {lat_threshold}°)")
    elif pole == 'south':
        filtered = crater_data[crater_data['lat'] <= -lat_threshold].copy()
        print(f"Filtered to {len(filtered)} South polar craters (lat <= -{lat_threshold}°)")
    elif pole == 'both':
        filtered = crater_data[np.abs(crater_data['lat']) >= lat_threshold].copy()
        print(f"Filtered to {len(filtered)} polar craters (|lat| >= {lat_threshold}°)")
    else:
        raise ValueError(f"pole must be 'north', 'south', or 'both', got '{pole}'")
    
    return filtered