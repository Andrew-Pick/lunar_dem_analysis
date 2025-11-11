import sys
import os
import matplotlib.pyplot as plt
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from crater_analysis.io import read_crater_locations, read_dem
from crater_analysis.analysis import filter_craters, get_crater_profile, get_dem_snippet
from crater_analysis.plotting import plot_dem, plot_crater_locations
from crater_analysis.core import Crater


dem, metadata = read_dem('../data/dems/LDEM_60S_120M.JP2')

lat = 'lat'
lon = 'lon'
diam = 'diameter_m'

#craters = read_crater_locations('../data/catalogs/lunar_crater_database_robbins_2018.csv')
#craters = filter_craters(craters, pole='south', lat_threshold=60, circ_threshold=None, diam_min=8, diam_max=None, lat_col='LAT_CIRC_IMG', diam_col='DIAM_CIRC_IMG')

craters = read_crater_locations('../data/catalogs/moon_data.csv')
craters = filter_craters(craters, pole='south', lat_threshold=60, circ_threshold=0.85, diam_min=8000, diam_max=None, lat_col=lat, diam_col=diam)

craters.head()

### Initialise Craters ###
# Create an empty list to hold the Crater objects
crater_objects = []

# Loop through the first few rows of the DataFrame for demonstration
# To process all craters, change `craters.head(1).iterrows()` to `craters.iterrows()`
for index, row in craters.head(100).iterrows():
    # Extract crater properties from the row
    crater_id = None #row['CRATER_ID']
    crater_lon = row[lon]
    crater_lat = row[lat]
    crater_diam_km = row[diam] #/1000.0  # Convert diameter from meters to kilometers
    
    # Get the DEM snippet for this crater
    dem_snippet, snippet_meta, center_coords = get_dem_snippet(
        dem,
        metadata,
        crater_lon,
        crater_lat,
        crater_diam_km
    )
    
    # Create the Crater object
    crater_obj = Crater(
        crater_id=crater_id,
        dem_snippet=dem_snippet,
        metadata=snippet_meta,
        center_coords=center_coords,
        global_coords={'lon': crater_lon, 'lat': crater_lat},
    )
    
    crater_obj.add_property('diameter_km', crater_diam_km)

    # Add the new object to our list
    crater_objects.append(crater_obj)

# Print a summary to confirm it worked
print(f"Successfully initialized {len(crater_objects)} Crater objects.")
if crater_objects:
    print("First crater object in the list:")
    print(crater_objects[0])
    print(crater_objects[0].dem_data)
print(dem)

plot_dem(crater_objects[50].dem_data, crater_objects[0].metadata)

# Extract profiles in both horizontal and vertical directions
crater = crater_objects[50]

# Horizontal profile (E-W, angle=0)
distance_h, elevation_h, pixel_coords_h = get_crater_profile(
    crater.dem_data,
    crater.metadata,
    crater.global_coords['lon'],
    crater.global_coords['lat'],
    crater_diam_km=crater.get_property('diameter_km') * 1.5,
    angle=0  # 0 degrees = horizontal (E-W)
)

# Vertical profile (N-S, angle=90)
distance_v, elevation_v, pixel_coords_v = get_crater_profile(
    crater.dem_data,
    crater.metadata,
    crater.global_coords['lon'],
    crater.global_coords['lat'],
    crater_diam_km=crater.get_property('diameter_km') * 1.5,
    angle=90  # 90 degrees = vertical (N-S)
)

print(f"Horizontal profile: {len(elevation_h)} pixels")
print(f"Vertical profile: {len(elevation_v)} pixels")

# Plot both profiles
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Horizontal profile
ax1.plot(distance_h, elevation_h, '-', linewidth=1, color='blue')
ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Crater center')
ax1.set_xlabel('Distance from center (km)')
ax1.set_ylabel('Elevation (m)')
ax1.set_title(f'Horizontal Profile (E-W)')
#ax1.grid(True, alpha=0.3)
ax1.legend()

# Vertical profile
ax2.plot(distance_v, elevation_v, '-', linewidth=1, color='green')
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Crater center')
ax2.set_xlabel('Distance from center (km)')
ax2.set_ylabel('Elevation (m)')
ax2.set_title(f'Vertical Profile (N-S)')
#ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()