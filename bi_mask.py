import rasterio
import numpy as np

dnbr_path = r"output_data\dnbr_utm.tif"
output_path = r"output_data\binäre_brandmaske_044.tif"


with rasterio.open(dnbr_path) as src:
    dnbr = src.read(1)
    meta = src.meta

bi_mask = np.zeros(dnbr.shape, dtype=np.uint8) #nparray mit 0 für alle werte wird initialisiert 
bi_mask[dnbr >= 0.44] = 1 #hier Grenzwert eingeben, überdem werte auf 1 gesetzt wird 

with rasterio.open(output_path, "w", **meta) as mask:
    mask.write(bi_mask, 1)