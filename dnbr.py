import rasterio
import numpy as np

nbr_pre_path = r"output_data\nbr_pre_utm.tif"
nbr_post_path = r"output_data\nbr_post_utm.tif"
out_path = r"output_data\dnbr_utm.tif"

with rasterio.open(nbr_pre_path) as pre_src:
    nbr_pre = pre_src.read(1).astype(np.float32)
    meta = pre_src.meta

with rasterio.open(nbr_post_path) as post_src:
    nbr_post = post_src.read(1).astype(np.float32)

dnbr = nbr_pre - nbr_post

with rasterio.open(out_path, "w", **meta) as dnbr_out:
    dnbr_out.write(dnbr, 1)
