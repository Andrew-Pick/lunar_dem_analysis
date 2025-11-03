import rasterio
import pandas as pd
from typing import List
from .core import Crater  # Assuming your Crater class is in core.py

def read_dem(file_path: str) -> tuple:
    """
    Reads a DEM file (like a GeoTIFF) and returns its data and metadata.

    Parameters:
    - file_path (str): The path to the DEM file.

    Returns:
    - tuple: A tuple containing:
        - dem_data (np.array): The 2D array of elevation data.
        - metadata (dict): The DEM's metadata, including transform for coordinates.
    """
    try:
        with rasterio.open(file_path) as src:
            dem_data = src.read(1)  # Read the first band
            metadata = src.profile
        print(f"Successfully read DEM from {file_path}")
        return dem_data, metadata
    except rasterio.errors.RasterioIOError as e:
        print(f"Error: Could not read DEM file at {file_path}")
        print(f"Details: {e}")
        return None, None

def read_crater_locations(file_path: str) -> pd.DataFrame:
    """
    Reads crater location data from a CSV or Excel file into a pandas DataFrame.

    The file should contain columns for crater ID and location (e.g., 'id', 'x', 'y').

    Parameters:
    - file_path (str): The path to the CSV or Excel file.

    Returns:
    - pd.DataFrame: A DataFrame containing the crater data, or None if an error occurs.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            # For excel files, you may need to install the 'openpyxl' package:
            # pip install openpyxl
            df = pd.read_excel(file_path)
        else:
            print(f"Error: Unsupported file format for {file_path}. Please use CSV or Excel.")
            return None
        
        print(f"Successfully read {len(df)} crater locations from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None