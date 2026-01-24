# DER WORKFLOW AUS SESSION 6 VOM ANDY, MUSS NOCH ANGEPASST WERDEN


# imports
from tempfile import template
import numpy as np
import rasterio
import rasterio.warp
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


###### STACK BANDS ######

# helper function for reading bands
def read_band(path_to_img):
    with rasterio.open(path_to_img, "r") as img:
        return img.read(1).astype(np.float32)

s2_bands = Path(r"C:\Users\felix\Documents\wald\post_utm\resampled")

bands = []
for band in s2_bands.glob("*.tiff"):
    data = read_band(band)
    bands.append(data)

bands = np.dstack(bands)

###### STACK BANDS ENDE ######

###### LABEL BLOCK (RESAMPLE TO BAND GRID) ###### Hier mal fragen ob das Label resampeln Problematisch ist, und wir die Labels nochmal neu anlegen sollten

label_raster_path = Path(r"C:\Users\felix\Documents\wald\output_data\Poly_manuell.tif")

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

###### LABEL BLOCK ENDE ######


####### Geometrie Check ######

print("Bands shape:", X.shape)
print("Labels shape:", y.shape)

####### Geometrie Check Ende ######

####### Prepare Data for ML, reshape X and y ######

rows, cols, bands = X.shape
X = X.reshape((rows * cols, bands))
y = y.reshape((rows * cols,))

print("X shape after reshape:", X.shape)
print("y shape after reshape:", y.shape)

# No Data Werte entfernen
valid_mask = (y == 0) | (y == 1)

# select only valid samples for training
X_train = X[valid_mask]
y_train = y[valid_mask]

print("Training samples:", X_train.shape[0])
print("Unique training labels:", np.unique(y_train))


######## Prepare Data for ML Ende ######


####### ML Training Block ######

# set up random forest
n_trees = 100
rf = RF(n_estimators=n_trees, n_jobs=-1, oob_score=True, random_state=123)

# train random forest
rf.fit(X_train, y_train)

print(rf.oob_score_)

# # # predict on all pixels (including those without a label/not used for training)
# # # here, we actually make use of the model we just trained!
# # # this step is also called "inference"

print("done.")




