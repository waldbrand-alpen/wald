import os
import rasterio


######## file Eigenschaften #########

out_format = "GTiff"
datatype_out = "Int16"
burn_field = "gridcode"   # name of the attribute column in the vector dataset
nodata_value = "-1"

######## file Eigenschaften ENDE #########

######## PFADE #########

# path to vector dataset (polygons)
training_vector = r"C:\Users\felix\Documents\wald\output_data\Polygone_burned_unburned.shp"

# path to template raster (resampled B02!)
template_raster = r"C:\Users\felix\Documents\wald\post_utm\resampled\rs_2024-09-05-00_00_2024-09-05-23_59_Sentinel-2_L2A_B02_(Raw).tiff"

# path to rasterized output
training_raster = r"C:\Users\felix\Documents\wald\output_data\labels_burned_unburned.tiff"

######## PFADE ENDE #########

######## Auflösung vom template Raster holen #########

with rasterio.open(template_raster) as src:
    width = src.width
    height = src.height
    transform = src.transform

xmin = transform.c
ymax = transform.f
xmax = xmin + width * transform.a
ymin = ymax + height * transform.e


######## Auflösuung vom template Raster holen ENDE #########

######## Rasterize Labels #########

cmd = (
    f"gdal_rasterize "
    f"-a {burn_field} "
    f"-of {out_format} "
    f"-te {xmin} {ymin} {xmax} {ymax} "
    f"-ts {width} {height} "
    f"-a_nodata -1 "
    f"-ot {datatype_out} "
    f"{training_vector} "
    f"{training_raster}"
)

print(cmd)
os.system(cmd)


with rasterio.open(template_raster) as src:
    print("Template raster shape:", src.shape)

with rasterio.open(training_raster) as src:
    print("Training raster shape:", src.shape)