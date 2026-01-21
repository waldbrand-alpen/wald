# DER WORKFLOW AUS SESSION 6 VOM ANDY, MUSS NOCH ANGEPASST WERDEN


# imports
import numpy as np
import rasterio
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as RF


# helper function for reading bands
def read_band(path_to_img):
    with rasterio.open(path_to_img, "r") as img:
        return img.read(1).astype(np.float32)


# define path to Sentinel-2 data directory and training dataset
s2_bands = Path(r"C:\0_Python\Python_2\6_ml_classification\data\data\s2_img")


# read individual Sentinel-2 bands as numpy array and add to a list
bands = []
for band in s2_bands.glob("*.tif"):
    data = read_band(band)
    bands.append(data)

# stack all bands along the third dimension in a numpy array
bands = np.dstack(bands)
# print (s2_data)

# read rasterized labels 
y = read_band(r"C:\0_Python\Python_2\6_ml_classification\data\data\output\output.tif")
# print(y.shape)

# reshape everything to one and two dimensional arrays


# make labels 1-dimensional


# extract number of rows, cols and bands from Sentinel-2 array for reshaping
rows, cols, n_bands = bands.shape

# reshape features (bands)
X = bands.reshape((rows * cols, n_bands))
y = y.reshape((rows * cols))

# print(X.shape)
# print(y.shape) 

# eliminate no-data pixels from both S2 array and labels (no data value is -1)
y_clean = y[y >= 0]

X_clean = X[y >= 0, :]

# check that the shape of the cleaned data sets matches
# print(X_clean.shape)
# print(y_clean.shape)

# create random forest model
n_trees = 100
rf = RF(n_estimators=n_trees, n_jobs=-1, oob_score=True, random_state=123)

# train random forest
rf.fit(X_clean, y_clean)

# predict on all pixels (including those without a label/not used for training)
# here, we actually make use of the model we just trained!
# this step is also called "inference"
y_predicted = rf.predict(X)

print(X.shape)
print(y_predicted.shape)

# now we need to reshape the output back to a 2-dimensional raster,
# otherwise we won't be able to view the output in QGIS!
y_predicted_2d  = y_predicted.reshape((rows, cols))
print(y_predicted_2d.shape)

# read metadata of label raster for output
template = {}
training_raster = r"C:\0_Python\Python_2\6_ml_classification\data\data\output\output.tif"
with rasterio.open(training_raster, "r") as img:
    template["crs"] = img.crs
    template["transform"] = img.transform
    template["height"] = img.height
    template["width"] = img.width


# write our predicted output as GeoTiff
with rasterio.open(
    "predicted_labels.tif",
    "w",
    driver="GTiff",
    crs=template["crs"],
    transform=template["transform"],
    width=template["width"],
    height=template["height"],
    count=1,
    dtype=y_predicted_2d.dtype,
) as fobj:
    fobj.write(y_predicted_2d, 1)

# now visualize the result in QGIS!
