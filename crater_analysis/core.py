import numpy as np

class Crater:
    """
    A class to represent a single lunar crater.

    This class acts as a container for the crater's location, the relevant
    DEM data, and the properties derived from analysis.
    """
    def __init__(self, crater_id, dem_snippet, center_coords=None, global_coords=None):
        """
        Initializes a Crater instance.

        Parameters:
        - crater_id (str or int): A unique identifier for the crater.
        - dem_snippet (np.array): A 2D NumPy array containing the DEM data
                                  for the area around the crater.
        - center_coords (tuple): The (row, col) coordinates of the crater's
                                 center within the dem_snippet.
        - global_coords (dict): A dictionary containing the global coordinates
                                (e.g., {'lon': 123.4, 'lat': -85.2}).
        """
        self.id = crater_id
        self.dem_data = dem_snippet
        self.center = center_coords if center_coords is not None else {}
        self.global_coords = global_coords if global_coords is not None else {}
        self.properties = {}  # A dictionary to store calculated properties

    def add_property(self, name, value):
        """
        Adds a calculated property to the crater's property dictionary.
        This provides a standard way to add new analysis results.

        Parameters:
        - name (str): The name of the property (e.g., 'depth', 'diameter').
        - value (any): The value of the property.
        """
        self.properties[name] = value

    def get_property(self, name):
        """
        Retrieves a property by its name.

        Parameters:
        - name (str): The name of the property to retrieve.

        Returns:
        - The value of the property, or None if it doesn't exist.
        """
        return self.properties.get(name)

    def __repr__(self):
        """
        Provides a developer-friendly string representation of the Crater.
        """
        return (f"Crater(id={self.id}, "
                f"center={self.center}, "
                f"global_coords={self.global_coords}, "
                f"properties={list(self.properties.keys())})")
