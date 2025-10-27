import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_raster(input_path, output_path, lat_0):

    moon_polar_stereo = {
        'proj': 'stere',
        'lat_0': lat_0,
        'lon_0': 0,
        'a': 1737400,
        'b': 1737400,
        'units': 'm'
    }

    target_crs = {
        'proj': 'eqc',
        'lon_0': 0,
        'lat_ts': 0,
        'a': 1737400,
        'b': 1737400,
        'units': 'm'
    }

    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            moon_polar_stereo, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        print("OG CRS:", src.crs)

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=moon_polar_stereo,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
            print("New CRS:", dst.crs)

    with rasterio.open(output_path) as ds:
        plt.imshow(ds.read(1), cmap='gray')
        plt.title("Reprojected to Equirectangular (Lunar)")
        plt.show()

    with rasterio.open(input_path) as ds:
        plt.imshow(ds.read(1), cmap='gray')
        plt.show()

path_flat = 'test.JP2'
path_3D = 'test_3D.JP2'
reproject_raster(path_flat, path_3D, lat_0=90)  #lat_0=90 for north pole and lat_0=-90 for south pole