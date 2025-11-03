import sys
import os
# Add parent directory to path so crater_analysis can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Plot depth/diameter vs latitude
from crater_analysis.analysis import filter_craters
from crater_analysis.plotting import plot_depth_diameter_vs_latitude
from crater_analysis.io import read_crater_locations

# Load crater data
crater_data = read_crater_locations('/home/andrew/lunar_dem_analysis/data/catalogs/moon_data.csv')

# Filter craters for the southern polar region
south_craters = filter_craters(crater_data, pole='south', lat_threshold=60)

# Filter craters for the northern polar region
north_craters = filter_craters(crater_data, pole='north', lat_threshold=60) 

# Plot with both scatter and binned averages
fig, ax = plot_depth_diameter_vs_latitude(north_craters, bins=10, show_scatter=True, show_binned=True)