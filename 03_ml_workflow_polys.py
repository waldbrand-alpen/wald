# DER WORKFLOW AUS SESSION 6 VOM ANDY, MUSS NOCH ANGEPASST WERDEN


# imports
import matplotlib.pyplot as plt
from tempfile import template
import numpy as np
import rasterio
import rasterio.warp
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


RANDOM_STATE = 42

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

# print(bands)
bands = np.dstack(bands)
print("Bänderformat:", bands.shape)

###### STACK BANDS ENDE ######

###### LABEL BLOCK ######

y = read_band(r"C:\Users\felix\Documents\wald\output_data\labels_burned_unburned.tiff")
print("Labelformat", y.shape)


####### LABEL BLOCK ENDE ######


# ####### Reshaping ######

rows, cols, n_bands = bands.shape

X = bands.reshape((rows * cols, n_bands))
y = y.reshape((rows * cols,))

print("X shape after reshape:", X.shape)
print("y shape after reshape:", y.shape)

####### Reshaping ENDE ######

####### No-Data bearbeiten ######

# eliminate no-data pixels from both S2 array and labels (no data value is -1)
y_clean = y[y >= 0]
X_clean = X[y >= 0, :]

# check that the shape of the cleaned data sets matches
print(X_clean.shape)
print(y_clean.shape)

# ####### No-Data bearbeiten ENDE ######

####### Split Daten für ML und RF ######

# extract train and test data and labels
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=RANDOM_STATE)

n_trees = 300
rf = RF(n_estimators=n_trees, n_jobs=-1, oob_score=True, random_state=RANDOM_STATE)

rf.fit(X_train, y_train)

# apply model on unseen test set
y_pred = rf.predict(X_test)

# calculate confusion matrix
cnf_mat = confusion_matrix(y_test, y_pred)
print(cnf_mat)
print("OOB score:", rf.oob_score_)

# plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_mat)
disp.plot()
plt.show() 

####### Split Daten für ML und RF ENDE ######

####### Predict on full image and GEO Output ######

y_full_pred = rf.predict(X_clean)

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
