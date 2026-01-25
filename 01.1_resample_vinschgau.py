##### Imports ######

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from pathlib import Path

##### Pfade ######

input_dir_reference = Path(r"C:\Users\felix\Documents\wald\post_utm\resampled")
input_dir = Path(r"C:\Users\felix\Documents\wald\vinschgau")
output_dir = Path(r"C:\Users\felix\Documents\wald\vinschgau\resampled")


##### Referenzband laden (NUR Pixelgröße von B02!) ######

ref_band = [f for f in input_dir_reference.glob("*.tiff") if "B02" in f.name][0]

with rasterio.open(ref_band) as ref:
    target_res = ref.res[0]    # <<< NUR Pixelgröße

##### Resample Bänder (nur Auflösung!) ######

band_files_vinschgau = list(input_dir.glob("*.tiff"))

for band_path in band_files_vinschgau:

    out_name = "rs_" + band_path.name
    out_path = output_dir / out_name 

    with rasterio.open(band_path) as src:
        src_data = src.read(1).astype(np.float32)

        # alte Bounds beibehalten
        left, bottom, right, top = src.bounds

        # neue Rastergröße aus neuer Auflösung berechnen
        new_width = int((right - left) / target_res)
        new_height = int((top - bottom) / target_res)

        # neuer Transform: gleiche Bounds, neue Pixelgröße
        dst_transform = from_bounds(
            left, bottom, right, top,
            new_width, new_height
        )

        dst_data = np.empty((new_height, new_width), dtype=np.float32)

        rasterio.warp.reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,          # <<< CRS bleibt identisch
            resampling=Resampling.bilinear
        )

        out_profile = src.profile.copy()
        out_profile.update(
            transform=dst_transform,
            height=new_height,
            width=new_width,
            dtype="float32",
            count=1
        )

        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(dst_data, 1)

print("resampling done (pixel size only).")



# Referenz: B02 (Jasper)
ref_b02 = r"C:\Users\felix\Documents\wald\post_utm\resampled\rs_2024-09-05-00_00_2024-09-05-23_59_Sentinel-2_L2A_B02_(Raw).tiff"

# Resampelte Vinschgau-Bänder
vinschgau_dir = Path(r"C:\Users\felix\Documents\wald\vinschgau\resampled")

with rasterio.open(ref_b02) as ref:
    ref_res = ref.res

print("B02 Pixelgröße:", ref_res)

for band in vinschgau_dir.glob("rs_*.tiff"):
    with rasterio.open(band) as src:
        print(f"{band.name} → Pixelgröße: {src.res}")