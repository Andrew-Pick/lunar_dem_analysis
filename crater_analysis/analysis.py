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
    pole='south', 
    lat_threshold=60, 
    circ_threshold=None, 
    diam_min=None, 
    diam_max=None,
    lat_col='lat',
    diam_col='diameter_m',
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


def smooth_profile(elevation, sigma=3):
    """
    Apply Gaussian smoothing to remove small-scale topographic noise from a profile.
    
    Uses a 1D Gaussian filter to smooth the elevation data, which is useful for
    removing high-frequency noise while preserving the overall crater morphology.
    
    Parameters:
    - elevation (np.array): Elevation values along the profile (e.g., in meters)
    - sigma (float): Standard deviation of the Gaussian kernel in pixels/samples.
                     Larger values = more smoothing (default: 3)
                     Recommended range: 1-5
    
    Returns:
    - np.array: Smoothed elevation profile
    
    Example:
    >>> elevation = np.array([100, 105, 102, 108, 103])  # Noisy data
    >>> smoothed = smooth_profile(elevation, sigma=2)
    >>> # smoothed will have reduced high-frequency variations
    
    Notes:
    - sigma=1-2: Light smoothing, preserves most detail
    - sigma=3-5: Moderate smoothing, good for typical noise
    - sigma>5: Heavy smoothing, may over-smooth real features
    - Apply smoothing BEFORE detrending for best results
    """
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(elevation, sigma=sigma)


def find_crater_floor(distance, elevation):
    """
    Find the deepest point in a crater profile using the second derivative.
    
    Following the Rubanenko et al. method: after detrending and smoothing,
    calculate the second derivative to identify inflection points, then
    find the lowest inflection point as the crater floor.
    
    Uses finite difference approximation via numpy.gradient for numerical derivatives.
    
    Parameters:
    - distance (np.array): Distance along the profile (e.g., in km)
    - elevation (np.array): Elevation values along the profile (e.g., in meters)
                           Should already be detrended and smoothed
    
    Returns:
    - tuple: (floor_idx, floor_distance, floor_elevation, second_derivative)
        - floor_idx (int): Array index of the deepest point
        - floor_distance (float): Distance coordinate of the deepest point
        - floor_elevation (float): Elevation at the deepest point
        - second_derivative (np.array): The computed second derivative
    
    Example:
    >>> distance = np.linspace(-5, 5, 100)
    >>> elevation = -(distance**2)  # Parabola with minimum at center
    >>> idx, dist, elev, d2 = find_crater_floor(distance, elevation)
    >>> # idx will be near 50 (center), dist near 0
    
    Notes:
    - Finds zero-crossings of the second derivative (inflection points)
    - Returns the inflection point with the lowest elevation
    - Fallback: if no inflection points found, returns global minimum elevation
    - For best results, apply to detrended and smoothed profiles
    - Near boundaries, finite differences are less accurate
    """
    # Calculate first derivative using central differences
    first_derivative = np.gradient(elevation, distance)
    
    # Calculate second derivative using central differences
    second_derivative = np.gradient(first_derivative, distance)
    
    # Find zero-crossings of second derivative (inflection points)
    # Look for sign changes in the second derivative
    sign_changes = np.diff(np.sign(second_derivative))
    inflection_indices = np.where(sign_changes != 0)[0]
    
    if len(inflection_indices) > 0:
        # Find the inflection point with the lowest elevation
        floor_idx = inflection_indices[np.argmin(elevation[inflection_indices])]
    else:
        # Fallback: if no inflection points found, use global minimum
        floor_idx = np.argmin(elevation)
    
    floor_distance = distance[floor_idx]
    floor_elevation = elevation[floor_idx]
    
    return floor_idx, floor_distance, floor_elevation, second_derivative


def find_crater_floor_max_curv(distance, elevation):
    # Calculate first derivative using central differences
    first_derivative = np.gradient(elevation, distance)
    
    # Calculate second derivative using central differences
    second_derivative = np.gradient(first_derivative, distance)

    floor_idx = np.argmax(second_derivative)
    floor_distance = distance[floor_idx]
    floor_elevation = elevation[floor_idx]
    
    return floor_idx, floor_distance, floor_elevation, second_derivative


def find_crater_rims(distance, elevation, floor_idx, threshold_fraction=0.1):
    """
    Find crater rims using the Rubanenko et al. method.
    
    Starting from the crater floor, divide the profile into two parts (left and right).
    For each side, find the steepest slope, then locate where the slope decreases
    to a specified fraction of that maximum slope. This identifies the rim locations.
    
    Parameters:
    - distance (np.array): Distance along the profile (e.g., in km)
    - elevation (np.array): Elevation values along the profile (e.g., in meters)
                           Should already be detrended and smoothed
    - floor_idx (int): Array index of the crater floor (from find_crater_floor)
    - threshold_fraction (float): Fraction of steepest slope to use as threshold
                                 (default: 0.1 for 10%)
    
    Returns:
    - tuple: (left_rim_idx, right_rim_idx, left_rim_distance, right_rim_distance, first_derivative)
        - left_rim_idx (int): Array index of the left rim
        - right_rim_idx (int): Array index of the right rim
        - left_rim_distance (float): Distance coordinate of left rim
        - right_rim_distance (float): Distance coordinate of right rim
        - first_derivative (np.array): The computed first derivative
    
    Example:
    >>> floor_idx, _, _, _ = find_crater_floor(distance, elevation)
    >>> left_idx, right_idx, left_dist, right_dist, deriv = find_crater_rims(distance, elevation, floor_idx)
    >>> diameter = right_dist - left_dist
    
    Notes:
    - The first derivative is calculated using numpy.gradient (central differences)
    - Left side: looks for steepest negative slope (descending into crater)
    - Right side: looks for steepest positive slope (ascending from crater)
    - Rims are where slope magnitude drops below threshold_fraction * max_slope
    - For best results, apply to detrended and smoothed profiles
    """
    # Calculate first derivative using central differences
    first_derivative = np.gradient(elevation, distance)
    
    # Split profile into left (before floor) and right (after floor) sections
    left_section = slice(0, floor_idx + 1)
    right_section = slice(floor_idx, len(distance))
    
    # LEFT SIDE: Find steepest negative slope (going down into crater)
    left_slopes = first_derivative[left_section]
    left_steepest_idx_local = np.argmin(left_slopes)  # Local index in left section
    left_steepest_idx = left_steepest_idx_local  # Convert to global index
    left_steepest_slope = left_slopes[left_steepest_idx_local]
    left_threshold = threshold_fraction * abs(left_steepest_slope)
    
    # Search outward from the STEEPEST SLOPE point (not from floor)
    # This avoids picking the crater floor as the rim
    left_rim_idx = None
    for i in range(left_steepest_idx, -1, -1):
        if abs(first_derivative[i]) < left_threshold:
            left_rim_idx = i
            break
            
    if left_rim_idx is None:
        raise ValueError("Left rim not found: slope did not decrease below threshold.")
    
    # RIGHT SIDE: Find steepest positive slope (going up from crater)
    right_slopes = first_derivative[right_section]
    right_steepest_idx_local = np.argmax(right_slopes)  # Local index in right section
    right_steepest_idx = floor_idx + right_steepest_idx_local  # Convert to global index
    right_steepest_slope = right_slopes[right_steepest_idx_local]
    right_threshold = threshold_fraction * abs(right_steepest_slope)
    
    # Search outward from the STEEPEST SLOPE point (not from floor)
    right_rim_idx = None
    for i in range(right_steepest_idx, len(distance)):
        if abs(first_derivative[i]) < right_threshold:
            right_rim_idx = i
            break
            
    if right_rim_idx is None:
        raise ValueError("Right rim not found: slope did not decrease below threshold.")
    
    left_rim_distance = distance[left_rim_idx]
    right_rim_distance = distance[right_rim_idx]
    
    return left_rim_idx, right_rim_idx, left_rim_distance, right_rim_distance, first_derivative


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
    # Perform linear least squares fit using the new numpy.polynomial API
    # Polynomial.fit returns a Polynomial object for degree 1 (linear)
    poly = np.polynomial.Polynomial.fit(distance, elevation, deg=1)
    
    # Extract coefficients: poly.coef[0] is intercept, poly.coef[1] is slope
    intercept = poly.coef[0]
    slope = poly.coef[1]
    
    # Calculate the linear trend using the polynomial
    linear_trend = poly(distance)
    
    # Subtract the trend from the elevation
    detrended_elevation = elevation - linear_trend
    
    return detrended_elevation, slope, intercept


def detrend_profile_robust(distance, elevation, fraction=0.2):
    """
    Remove large-scale slope effects by fitting a trend to the profile edges only.
    This avoids the crater shape biasing the trend line.
    
    Parameters:
    - distance (np.array): Distance along the profile
    - elevation (np.array): Elevation values
    - fraction (float): Fraction of the profile at each end to use for fitting (default 0.2)
    
    Returns:
    - tuple: (detrended_elevation, slope, intercept)
    """
    n = len(distance)
    n_edge = int(n * fraction)
    
    # Select points from the start and end of the profile
    mask = np.zeros(n, dtype=bool)
    mask[:n_edge] = True
    mask[-n_edge:] = True
    
    dist_fit = distance[mask]
    elev_fit = elevation[mask]
    
    # Fit linear trend to these points
    poly = np.polynomial.Polynomial.fit(dist_fit, elev_fit, deg=1)
    
    intercept = poly.coef[0]
    slope = poly.coef[1]
    
    # Calculate trend for the whole profile
    linear_trend = poly(distance)
    
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


class CraterDepthAnalyzer:
    """
    Automated crater depth analysis pipeline following Rubanenko et al. methodology.
    
    This class encapsulates the entire workflow from loading data to generating
    d/D vs latitude plots comparing catalog and measured depths.
    
    Example:
    >>> from crater_analysis.io import read_dem, read_crater_locations
    >>> from crater_analysis.analysis import CraterDepthAnalyzer
    >>> 
    >>> # Load DEM and catalog
    >>> dem, metadata = read_dem('data/dems/LDEM_60S_120M.JP2')
    >>> craters = read_crater_locations('data/catalogs/moon_data.csv')
    >>> 
    >>> # Create analyzer
    >>> analyzer = CraterDepthAnalyzer(
    ...     dem=dem,
    ...     metadata=metadata,
    ...     catalog_data=craters,
    ...     pole='south',
    ...     lat_threshold=60,
    ...     lat_col='lat',
    ...     lon_col='lon',
    ...     diam_col='diameter_m',
    ...     depth_col='depth_m',
    ...     diam_is_meters=True
    ... )
    >>> 
    >>> # Run analysis and plot
    >>> analyzer.run_analysis()
    >>> analyzer.plot_d_D_vs_latitude()
    """
    
    def __init__(
        self,
        dem,
        metadata,
        catalog_data,
        pole='south',
        lat_threshold=60,
        circ_threshold=None,
        diam_min=None,
        diam_max=None,
        lat_col='lat',
        lon_col='lon',
        diam_col='diameter_m',
        depth_col='depth_m',
        diam_is_meters=True,
        has_depth=True,
        sigma=3,
        detrend_fraction=0.2,
        profile_factor=1.5,
        threshold_fraction=0.1,
        d_D_min=0.025,
        d_D_max=0.25
    ):
        """
        Initialize the CraterDepthAnalyzer.
        
        Parameters:
        - dem (np.array): DEM elevation data
        - metadata (dict): DEM metadata with transform and CRS
        - catalog_data (pd.DataFrame): Crater catalog
        - pole (str): 'north' or 'south'
        - lat_threshold (float): Latitude threshold for filtering
        - circ_threshold (float): Circularity threshold (None to skip)
        - diam_min (float): Minimum diameter (in units matching diam_is_meters)
        - diam_max (float): Maximum diameter (in units matching diam_is_meters)
        - lat_col (str): Name of latitude column
        - lon_col (str): Name of longitude column
        - diam_col (str): Name of diameter column
        - depth_col (str): Name of depth column (ignored if has_depth=False)
        - diam_is_meters (bool): True if diameter is in meters, False if km
        - has_depth (bool): True if catalog has depth data
        - sigma (float): Gaussian smoothing parameter
        - detrend_fraction (float): Fraction of profile edges for detrending
        - profile_factor (float): Multiplier for profile length (default 1.5x diameter)
        - threshold_fraction (float): Rim detection threshold
        - d_D_min (float): Minimum d/D ratio for quality filter
        - d_D_max (float): Maximum d/D ratio for quality filter
        """
        self.dem = dem
        self.metadata = metadata
        self.pole = pole
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.diam_col = diam_col
        self.depth_col = depth_col
        self.has_depth = has_depth
        self.diam_is_meters = diam_is_meters
        
        # Processing parameters
        self.sigma = sigma
        self.detrend_fraction = detrend_fraction
        self.profile_factor = profile_factor
        self.threshold_fraction = threshold_fraction
        self.d_D_min = d_D_min
        self.d_D_max = d_D_max
        
        # Filter craters
        self.craters = filter_craters(
            catalog_data,
            pole=pole,
            lat_threshold=lat_threshold,
            circ_threshold=circ_threshold,
            diam_min=diam_min,
            diam_max=diam_max,
            lat_col=lat_col,
            diam_col=diam_col
        )
        
        # Results storage
        self.results = {
            'measured_depth': [],
            'catalog_depth': [] if has_depth else None,
            'catalog_diameter': [],
            'measured_diameter': [],
            'latitude': [],
            'longitude': [],
            'crater_idx': [],
            'error_craters': []
        }
        
    def run_analysis(self, verbose=True):
        """
        Run the complete crater depth analysis pipeline.
        
        Parameters:
        - verbose (bool): Print progress updates
        
        Returns:
        - dict: Results dictionary with all measurements
        """
        if verbose:
            print(f"Processing {len(self.craters)} craters...")
        
        for idx, (_, row) in enumerate(self.craters.iterrows()):
            try:
                # Extract crater properties
                crater_lon = row[self.lon_col]
                crater_lat = row[self.lat_col]
                crater_diam_km = row[self.diam_col] / 1000.0 if self.diam_is_meters else row[self.diam_col]
                
                if self.has_depth:
                    catalog_depth = row[self.depth_col]
                
                # Get DEM snippet
                dem_snippet, snippet_meta, _ = get_dem_snippet(
                    self.dem, self.metadata,
                    crater_lon, crater_lat,
                    crater_diam_km,
                    padding_factor=1.5
                )
                
                # Extract profiles
                distance_h, elevation_h, _ = get_crater_profile(
                    dem_snippet, snippet_meta,
                    crater_lon, crater_lat,
                    crater_diam_km=crater_diam_km * self.profile_factor,
                    angle=0
                )
                distance_v, elevation_v, _ = get_crater_profile(
                    dem_snippet, snippet_meta,
                    crater_lon, crater_lat,
                    crater_diam_km=crater_diam_km * self.profile_factor,
                    angle=90
                )
                
                # Detrend and smooth
                elevation_h_detrended, _, _ = detrend_profile_robust(
                    distance_h, elevation_h, fraction=self.detrend_fraction
                )
                elevation_v_detrended, _, _ = detrend_profile_robust(
                    distance_v, elevation_v, fraction=self.detrend_fraction
                )
                elevation_h_smooth = smooth_profile(elevation_h_detrended, sigma=self.sigma)
                elevation_v_smooth = smooth_profile(elevation_v_detrended, sigma=self.sigma)
                
                # Find floor
                floor_idx_h, _, _, _ = find_crater_floor(distance_h, elevation_h_smooth)
                floor_idx_v, _, _, _ = find_crater_floor(distance_v, elevation_v_smooth)
                
                # Find rims
                left_rim_idx_h, right_rim_idx_h, left_rim_dist_h, right_rim_dist_h, _ = find_crater_rims(
                    distance_h, elevation_h_smooth, floor_idx_h, 
                    threshold_fraction=self.threshold_fraction
                )
                left_rim_idx_v, right_rim_idx_v, left_rim_dist_v, right_rim_dist_v, _ = find_crater_rims(
                    distance_v, elevation_v_smooth, floor_idx_v,
                    threshold_fraction=self.threshold_fraction
                )
                
                # Calculate depths
                center_idx_h = (left_rim_idx_h + right_rim_idx_h) // 2
                avg_rim_elev_h = (elevation_h_smooth[left_rim_idx_h] + elevation_h_smooth[right_rim_idx_h]) / 2
                depth_h = avg_rim_elev_h - elevation_h_smooth[center_idx_h]
                
                center_idx_v = (left_rim_idx_v + right_rim_idx_v) // 2
                avg_rim_elev_v = (elevation_v_smooth[left_rim_idx_v] + elevation_v_smooth[right_rim_idx_v]) / 2
                depth_v = avg_rim_elev_v - elevation_v_smooth[center_idx_v]
                
                avg_depth = (depth_h + depth_v) / 2
                
                # Calculate diameters
                diameter_h = right_rim_dist_h - left_rim_dist_h
                diameter_v = right_rim_dist_v - left_rim_dist_v
                avg_diameter = (diameter_h + diameter_v) / 2
                
                # Quality filter: reasonable d/D ratio
                d_D_ratio = avg_depth / (avg_diameter * 1000)
                if self.d_D_min < d_D_ratio < self.d_D_max:
                    self.results['measured_depth'].append(avg_depth)
                    if self.has_depth:
                        self.results['catalog_depth'].append(catalog_depth)
                    self.results['catalog_diameter'].append(crater_diam_km)
                    self.results['measured_diameter'].append(avg_diameter)
                    self.results['latitude'].append(crater_lat)
                    self.results['longitude'].append(crater_lon)
                    self.results['crater_idx'].append(idx)
                
            except (ValueError, IndexError) as e:
                self.results['error_craters'].append((idx, str(e)))
                continue
            
            if verbose and (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(self.craters)} craters...")
        
        if verbose:
            print(f"\nSuccessfully processed {len(self.results['measured_depth'])} craters")
            print(f"Failed to process {len(self.results['error_craters'])} craters")
        
        return self.results
    
    def plot_d_D_vs_latitude(
        self,
        n_bins=10,
        figsize=(10, 6),
        ylim=(0.03, 0.2),
        show_failed=False,
        save_path=None
    ):
        """
        Plot d/D ratio vs latitude with catalog and measured data on the same graph.
        
        Parameters:
        - n_bins (int): Number of latitude bins for averaging
        - figsize (tuple): Figure size (width, height)
        - ylim (tuple): Y-axis limits for d/D ratio
        - show_failed (bool): Show failed craters as red X markers
        - save_path (str): Path to save figure (None = don't save)
        
        Returns:
        - matplotlib.figure.Figure: The created figure
        """
        import matplotlib.pyplot as plt
        
        if len(self.results['measured_depth']) == 0:
            raise ValueError("No results to plot. Run run_analysis() first.")
        
        # Convert to arrays
        measured_depths = np.array(self.results['measured_depth'])
        measured_diameters = np.array(self.results['measured_diameter'])
        catalog_diameters = np.array(self.results['catalog_diameter'])
        latitudes = np.array(self.results['latitude'])
        
        # Calculate d/D ratios
        d_D_ratio_measured = measured_depths / (measured_diameters * 1000)
        
        # Use absolute latitude
        abs_latitudes = np.abs(latitudes)
        
        # Create DataFrame for binning
        df = pd.DataFrame({
            'lat': abs_latitudes,
            'd_D_measured': d_D_ratio_measured
        })
        
        if self.has_depth:
            catalog_depths = np.array(self.results['catalog_depth'])
            d_D_ratio_catalog = catalog_depths / (catalog_diameters * 1000)
            df['d_D_catalog'] = d_D_ratio_catalog
        
        # Binning
        lat_bins = np.linspace(df['lat'].min(), df['lat'].max(), n_bins + 1)
        bin_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
        
        # Calculate binned statistics for measured data
        measured_means = []
        measured_sems = []
        for i in range(len(lat_bins) - 1):
            mask = (df['lat'] >= lat_bins[i]) & (df['lat'] < lat_bins[i+1])
            bin_data = df[mask]['d_D_measured']
            if len(bin_data) > 0:
                measured_means.append(bin_data.mean())
                measured_sems.append(bin_data.std() / np.sqrt(len(bin_data)))
            else:
                measured_means.append(np.nan)
                measured_sems.append(np.nan)
        
        measured_means = np.array(measured_means)
        measured_sems = np.array(measured_sems)
        
        # Calculate binned statistics for catalog data if available
        if self.has_depth:
            catalog_means = []
            catalog_sems = []
            for i in range(len(lat_bins) - 1):
                mask = (df['lat'] >= lat_bins[i]) & (df['lat'] < lat_bins[i+1])
                bin_data = df[mask]['d_D_catalog']
                if len(bin_data) > 0:
                    catalog_means.append(bin_data.mean())
                    catalog_sems.append(bin_data.std() / np.sqrt(len(bin_data)))
                else:
                    catalog_means.append(np.nan)
                    catalog_sems.append(np.nan)
            
            catalog_means = np.array(catalog_means)
            catalog_sems = np.array(catalog_sems)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot individual points (semi-transparent)
        if self.has_depth:
            ax.scatter(abs_latitudes, d_D_ratio_measured, alpha=0.2, s=10, 
                      color='blue', label='_nolegend_')
            ax.scatter(abs_latitudes, d_D_ratio_catalog, alpha=0.2, s=10,
                      color='orange', label='_nolegend_')
        else:
            ax.scatter(abs_latitudes, d_D_ratio_measured, alpha=0.3, s=10,
                      color='blue', label='Measured')
        
        # Plot failed craters if requested
        if show_failed and len(self.results['error_craters']) > 0:
            failed_indices = [idx for idx, _ in self.results['error_craters']]
            failed_rows = self.craters.iloc[failed_indices]
            failed_lats = np.abs(failed_rows[self.lat_col].values)
            
            if self.has_depth:
                failed_depths = failed_rows[self.depth_col].values
                failed_diams = failed_rows[self.diam_col].values
                if self.diam_is_meters:
                    failed_diams = failed_diams / 1000.0
                failed_d_D = failed_depths / (failed_diams * 1000)
                ax.scatter(failed_lats, failed_d_D, alpha=0.5, s=20, color='red',
                          marker='x', label=f'Failed (n={len(failed_indices)})')
            else:
                # Just show X markers at a fixed y position if no depth data
                ax.scatter(failed_lats, np.full_like(failed_lats, ylim[0] + 0.01),
                          alpha=0.5, s=20, color='red', marker='x',
                          label=f'Failed (n={len(failed_indices)})')
        
        # Plot binned averages
        ax.errorbar(bin_centers, measured_means, yerr=measured_sems,
                   color='blue', linewidth=2, marker='o', markersize=6,
                   capsize=5, label='Measured (binned)')
        
        if self.has_depth:
            ax.errorbar(bin_centers, catalog_means, yerr=catalog_sems,
                       color='red', linewidth=2, marker='s', markersize=6,
                       capsize=5, label='Catalog (binned)')
        
        ax.set_xlabel('Latitude (degrees)', fontsize=12)
        ax.set_ylabel('Depth/Diameter Ratio', fontsize=12)
        
        title = f'd/D vs Latitude - {self.pole.capitalize()} Pole'
        if self.has_depth:
            title += ' (Measured vs Catalog)'
        ax.set_title(title, fontsize=14)
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(ylim)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def get_summary_statistics(self):
        """
        Get summary statistics for the analysis results.
        
        Returns:
        - dict: Dictionary containing summary statistics
        """
        if len(self.results['measured_depth']) == 0:
            raise ValueError("No results available. Run run_analysis() first.")
        
        measured_depths = np.array(self.results['measured_depth'])
        measured_diameters = np.array(self.results['measured_diameter'])
        catalog_diameters = np.array(self.results['catalog_diameter'])
        
        d_D_measured = measured_depths / (measured_diameters * 1000)
        
        stats = {
            'n_successful': len(measured_depths),
            'n_failed': len(self.results['error_craters']),
            'measured_depth': {
                'mean': np.mean(measured_depths),
                'median': np.median(measured_depths),
                'std': np.std(measured_depths),
                'min': np.min(measured_depths),
                'max': np.max(measured_depths)
            },
            'measured_diameter': {
                'mean': np.mean(measured_diameters),
                'median': np.median(measured_diameters),
                'std': np.std(measured_diameters),
                'min': np.min(measured_diameters),
                'max': np.max(measured_diameters)
            },
            'd_D_measured': {
                'mean': np.mean(d_D_measured),
                'median': np.median(d_D_measured),
                'std': np.std(d_D_measured),
                'min': np.min(d_D_measured),
                'max': np.max(d_D_measured)
            }
        }
        
        if self.has_depth:
            catalog_depths = np.array(self.results['catalog_depth'])
            d_D_catalog = catalog_depths / (catalog_diameters * 1000)
            depth_diff = measured_depths - catalog_depths
            diam_diff = measured_diameters - catalog_diameters
            
            stats['catalog_depth'] = {
                'mean': np.mean(catalog_depths),
                'median': np.median(catalog_depths),
                'std': np.std(catalog_depths),
                'min': np.min(catalog_depths),
                'max': np.max(catalog_depths)
            }
            stats['depth_difference'] = {
                'mean': np.mean(depth_diff),
                'median': np.median(depth_diff),
                'std': np.std(depth_diff),
                'min': np.min(depth_diff),
                'max': np.max(depth_diff)
            }
            stats['diameter_difference'] = {
                'mean': np.mean(diam_diff),
                'median': np.median(diam_diff),
                'std': np.std(diam_diff),
                'min': np.min(diam_diff),
                'max': np.max(diam_diff)
            }
            stats['depth_correlation'] = np.corrcoef(catalog_depths, measured_depths)[0, 1]
            stats['diameter_correlation'] = np.corrcoef(catalog_diameters, measured_diameters)[0, 1]
        
        return stats
    
    def print_summary(self):
        """Print a formatted summary of the analysis results."""
        stats = self.get_summary_statistics()
        
        print("\n" + "="*60)
        print("CRATER DEPTH ANALYSIS SUMMARY")
        print("="*60)
        print(f"\nProcessed: {stats['n_successful']} successful, {stats['n_failed']} failed")
        
        print(f"\nMeasured Depths:")
        print(f"  Mean: {stats['measured_depth']['mean']:.1f} m")
        print(f"  Median: {stats['measured_depth']['median']:.1f} m")
        print(f"  Std Dev: {stats['measured_depth']['std']:.1f} m")
        print(f"  Range: {stats['measured_depth']['min']:.1f} - {stats['measured_depth']['max']:.1f} m")
        
        print(f"\nMeasured Diameters:")
        print(f"  Mean: {stats['measured_diameter']['mean']:.2f} km")
        print(f"  Median: {stats['measured_diameter']['median']:.2f} km")
        print(f"  Std Dev: {stats['measured_diameter']['std']:.2f} km")
        print(f"  Range: {stats['measured_diameter']['min']:.2f} - {stats['measured_diameter']['max']:.2f} km")
        
        print(f"\nd/D Ratio (Measured):")
        print(f"  Mean: {stats['d_D_measured']['mean']:.4f}")
        print(f"  Median: {stats['d_D_measured']['median']:.4f}")
        print(f"  Std Dev: {stats['d_D_measured']['std']:.4f}")
        print(f"  Range: {stats['d_D_measured']['min']:.4f} - {stats['d_D_measured']['max']:.4f}")
        
        if self.has_depth:
            print(f"\nCatalog Depths:")
            print(f"  Mean: {stats['catalog_depth']['mean']:.1f} m")
            print(f"  Median: {stats['catalog_depth']['median']:.1f} m")
            print(f"  Std Dev: {stats['catalog_depth']['std']:.1f} m")
            print(f"  Range: {stats['catalog_depth']['min']:.1f} - {stats['catalog_depth']['max']:.1f} m")
            
            print(f"\nDepth Correlation (Catalog vs Measured): {stats['depth_correlation']:.3f}")
            print(f"Diameter Correlation (Catalog vs Measured): {stats['diameter_correlation']:.3f}")