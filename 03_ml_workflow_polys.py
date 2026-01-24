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

# ####### Reshaping ENDE ######

# ####### No-Data bearbeiten ######

# eliminate no-data pixels from both S2 array and labels (no data value is -1)
y_clean = y[y >= 0]
X_clean = X[y >= 0, :]

# check that the shape of the cleaned data sets matches
print(X_clean.shape)
print(y_clean.shape)

# ####### No-Data bearbeiten ENDE ######



####### Prepare Data for ML, reshape X and y ######

# rows, cols, bands = X.shape
# X = X.reshape((rows * cols, bands))
# y = y.reshape((rows * cols,))

# print("X shape after reshape:", X.shape)
# print("y shape after reshape:", y.shape)

# # No Data Werte entfernen
# valid_mask = (y == 0) | (y == 1)

# # select only valid samples for training
# X_train = X[valid_mask]
# y_train = y[valid_mask]

# print("Training samples:", X_train.shape[0])
# print("Unique training labels:", np.unique(y_train))


# ######## Prepare Data for ML Ende ######






