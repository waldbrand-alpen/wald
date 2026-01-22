# DER WORKFLOW AUS SESSION 6 VOM ANDY, MUSS NOCH ANGEPASST WERDEN


# imports
from tempfile import template
import numpy as np
import rasterio
import rasterio.warp
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as RF


####### Resample Block Satellite Images ######## (Satellitenbilder auf gleiche Auflösung bringen)

# Pfad 
s2_dir = Path(r"C:\Users\felix\Documents\wald\post_utm")

# Alle Banddateien einsammeln
band_files = list(s2_dir.glob("*.tiff"))

if len(band_files) == 0:
    raise RuntimeError("No band files found!")

print("Found band files:")
for bf in band_files:
    print(f" - {bf.name}")

# Referenzband für Resampling auswählen (erstes Band) hier kann man diskutieren ob man lieber die 10m Bänder oder 20m Bänder als Referenz nimmt !!!!!!!!!!!!!!!!!
ref_band_path = [f for f in band_files if "B02" in f.name][0]  # Blaues Band (B02) als Referenz

# Referenz laden 
with rasterio.open(ref_band_path) as ref:
    ref_shape = ref.shape
    ref_transform = ref.transform
    ref_crs = ref.crs

# print(ref_band_path.name)
# print(ref_shape)

# Alle Bänder laden und resamplen (src = Quelle, dst = Ziel)
resampled_bands = []

for band_file in band_files:
    with rasterio.open(band_file) as src:
        band = src.read(1).astype(np.float32)

        # Fall 1: Band passt schon (umnötiges Resampling vermeiden)
        if src.shape == ref_shape and src.transform == ref_transform:
            resampled_bands.append(band)

        # Fall 2: Band resamplen
        else:
            dst = np.empty(ref_shape, dtype=np.float32)

            rasterio.warp.reproject(
                source=band,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=rasterio.enums.Resampling.bilinear
            )

            resampled_bands.append(dst)

print("Resampling done.")

####### Resample Block Ende ######## (Bänder sind in resampled_bands Liste)


###### STACK BANDS ######

# stack all bands along the third dimension in a numpy array
X = np.dstack(resampled_bands)
print("Stacking done.")

###### STACK BANDS ENDE ######

###### LABEL BLOCK (RESAMPLED TO BAND GRID) ###### Hier mal fragen ob das Label resampeln Problematisch ist, und wir die Labels nochmal neu anlegen sollten

label_raster_path = Path(
    r"C:\Users\felix\Documents\wald\output_data\Poly_manuell.tif")

with rasterio.open(label_raster_path) as src:

    # Falls geometrie schon passt
    if src.shape == ref_shape and src.transform == ref_transform:
        y = src.read(1).astype(np.int32)

    # Sonst: auch Polygone Resamplen
    else:
        y = np.empty(ref_shape, dtype=np.int32)

        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=y,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=rasterio.enums.Resampling.nearest)
        

print("Labels loaded.")
# print("Label shape:", y.shape)
print("Unique label values:", np.unique(y))


######## Label output checken ########## ??????

# # write our predicted output as GeoTiff
# with rasterio.open(template) as src:
#     with rasterio.open(
#         "predicted_labels.tif",
#         "w",
#         driver="GTiff",
#         crs=src.crs,
#         transform=src.transform,
#         width=src.width,
#         height=src.height,
#         count=1,
#         dtype=y.dtype,
#     ) as dst:
#         dst.write(y, 1)

###### LABEL BLOCK ENDE ######


####### Geometrie Check ######

if y.shape != X.shape[:2]:
    raise RuntimeError("Shape mismatch between labels and bands!")

print("Bands shape:", X.shape)
print("Labels shape:", y.shape)

####### Geometrie Check Ende ######

####### Prepare Data for ML, reshape X and y ######

rows, cols, bands = X.shape
X = X.reshape((rows * cols, bands))
y = y.reshape((rows * cols,))

print("X shape after reshape:", X.shape)
print("y shape after reshape:", y.shape)

######## Prepare Data for ML Ende ######


####### ML Training Block ######

# create random forest model
n_trees = 100
rf = RF(n_estimators=n_trees, n_jobs=-1, oob_score=True, random_state=123)


# train random forest
rf.fit(X, y)

# # # predict on all pixels (including those without a label/not used for training)
# # # here, we actually make use of the model we just trained!
# # # this step is also called "inference"


# # y_predicted = rf.predict(X)

# # print(X.shape)
# # print(y_predicted.shape)

# # # now we need to reshape the output back to a 2-dimensional raster,
# # # otherwise we won't be able to view the output in QGIS!
# # y_predicted_2d  = y_predicted.reshape((rows, cols))
# # print(y_predicted_2d.shape)

# # # read metadata of label raster for output !!!HIER PFAD ANPASSEN!!!
# # template = {}
# # training_raster = r"C:\Users\felix\Documents\wald\output_data\Poly_manuell.tif"
# # with rasterio.open(training_raster, "r") as img:
# #     template["crs"] = img.crs
# #     template["transform"] = img.transform
# #     template["height"] = img.height
# #     template["width"] = img.width


# # # write our predicted output as GeoTiff
# # with rasterio.open(
# #     "predicted_labels.tif",
# #     "w",
# #     driver="GTiff",
# #     crs=template["crs"],
# #     transform=template["transform"],
# #     width=template["width"],
# #     height=template["height"],
# #     count=1,
# #     dtype=y_predicted_2d.dtype,
# # ) as fobj:
# #     fobj.write(y_predicted_2d, 1)

# # # now visualize the result in QGIS!


# print ("done")