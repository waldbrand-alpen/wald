##### Imports ######

import numpy as np
import rasterio
import rasterio.warp
from pathlib import Path
from rasterio.enums import Resampling

##### Imports ENDE ######

##### Pfade ######

input_dir = Path(r"C:\Users\felix\Documents\wald\post_utm")

output_dir = Path(r"C:\Users\felix\Documents\wald\post_utm\resampled")

##### Pfade ENDE ######

##### Referenzband laden (B02) ######

band_files = list(input_dir.glob("*.tiff"))

ref_band = [f for f in band_files if "2024-09-05-00_00_2024-09-05-23_59_Sentinel-2_L2A_B02_(Raw)" in f.name][0]

with rasterio.open(ref_band) as ref:
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_shape = ref.shape
    ref_profile = ref.profile

##### Referenzband laden ENDE ######

##### Resample Bänder ######

for band_path in band_files:

    out_name = "rs_" + band_path.name
    out_path = output_dir / out_name 

    with rasterio.open(band_path) as src:
        src_data = src.read(1).astype(np.float32)

        dst_data = np.empty(ref_shape, dtype=np.float32)

        rasterio.warp.reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )

        out_profile = ref_profile.copy()
        out_profile.update(
            dtype="float32",
            count=1
        )


        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(dst_data, 1)


print("done.") 