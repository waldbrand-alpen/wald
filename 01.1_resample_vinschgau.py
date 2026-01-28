##### Imports ######

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
from pathlib import Path

##### Pfade ######

input_dir_reference = Path(r"C:\Users\felix\Documents\wald\post_utm\resampled")
input_dir = Path(r"C:\Users\felix\Documents\wald\vinschgau")
output_dir = Path(r"C:\Users\felix\Documents\wald\vinschgau\resampled")


##### die Pixelgröße des Referenzbands laden  ######

ref_band = [f for f in input_dir_reference.glob("*.tiff") if "B02" in f.name][0]

with rasterio.open(ref_band) as ref:
    target_res = ref.res

##### Resample nur die Pixelgröße der Bänder  ######

band_files_vinschgau = list(input_dir.glob("*.tiff"))

for band_path in band_files_vinschgau:

    out_name = "rs_" + band_path.name       # Neuer Output (rs_.......tiff)
    out_path = output_dir / out_name 

    with rasterio.open(band_path) as src:
        src_data = src.read(1).astype(np.float32)

        dst_transform, dst_width, dst_height = calculate_default_transform(     # Geometrie für Zeilraster
            src.crs,
            src.crs,        # nur Resampling der Auflösung, nicht reprojizieren
            src.width,
            src.height,
            *src.bounds,        # * packt die bounds-Tuple in einzelne Werte aus
            resolution=target_res
        )

        dst_data = np.empty((dst_height, dst_width), dtype=np.float32)

        rasterio.warp.reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear      # Bilinear wichtig bei kontinuierlichen Daten (hier Reflektanz)
        )

        out_profile = src.profile.copy()
        out_profile.update(
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            dtype="float32",
            count=1
        )

        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(dst_data, 1)

print("done.")



##### Überprüfung der Pixelgröße der resampleten Bänder ######

# Referenz: B02 (Jasper)
ref_b02 = r"C:\Users\felix\Documents\wald\post_utm\resampled\rs_2024-09-05-00_00_2024-09-05-23_59_Sentinel-2_L2A_B02_(Raw).tiff"

# Resampelte Vinschgau-Bänder
vinschgau_dir = Path(r"C:\Users\felix\Documents\wald\vinschgau\resampled")

with rasterio.open(ref_b02) as ref:
    ref_res = ref.res

print("B02 Pixelgröße:", ref_res)

for band in vinschgau_dir.glob("rs_*.tiff"):
    with rasterio.open(band) as src:
        print(f"{band.name} = Pixelgröße: {src.res}")

print("done.")