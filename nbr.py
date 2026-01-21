import rasterio
import numpy as np


# Eingabepfade

nir_path_8a = r"pre_utm\2024-07-17-00_00_2024-07-17-23_59_Sentinel-2_L2A_B8A_(Raw).tiff"
swir_path_12 = r"pre_utm\2024-07-17-00_00_2024-07-17-23_59_Sentinel-2_L2A_B12_(Raw).tiff"

out_path = r"output_data\nbr_pre_utm.tif"


# Bänder laden

with rasterio.open(nir_path_8a) as nir_src:
    nir = nir_src.read(1).astype(np.float32)
    meta = nir_src.meta.copy()
    nir_transform = nir_src.transform
    nir_crs = nir_src.crs
    nir_shape = (nir_src.height, nir_src.width)

with rasterio.open(swir_path_12) as swir_src:
    swir = swir_src.read(1).astype(np.float32)
    swir_transform = swir_src.transform
    swir_crs = swir_src.crs
    swir_shape = (swir_src.height, swir_src.width)


# Schauen ob Daten im gleichen Format vorliegen

if nir_crs != swir_crs:
    raise ValueError("CRS!")

if nir_transform != swir_transform:
    raise ValueError("Transform!")

if nir_shape != swir_shape:
    raise ValueError("Rastergröße!")


# NBR berechnen

nenner = nir + swir
nbr = np.full(nir.shape, np.nan, dtype=np.float32) #Ergebnisarray initialisieren, gleiche Dimension, alle Werte nan

valid = nenner != 0 #checken ob nenner = 0 

nbr[valid] = (nir[valid] - swir[valid]) / nenner[valid] #Boolean Indexing, nimmt nur werte wo nenner nicht 0 ist 


# Metadaten & Ausgabe

meta.update(
    dtype="float32",
    count=1,
    nodata=np.nan,
    driver="GTiff"
)

with rasterio.open(out_path, "w", **meta) as nbr_out:
    nbr_out.write(nbr, 1)
