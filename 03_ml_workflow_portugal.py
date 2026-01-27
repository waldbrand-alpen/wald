# # DER WORKFLOW AUS SESSION 6 VOM ANDY, MUSS NOCH ANGEPASST WERDEN


# # imports
import matplotlib.pyplot as plt
from tempfile import template
import numpy as np
import rasterio
import rasterio.warp
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# RANDOM_STATE = 42


# ###### STACK BANDS Jasper ######

# # helper function for reading bands
# def read_band(path_to_img):
#     with rasterio.open(path_to_img, "r") as img:
#         return img.read(1).astype(np.float32)

# # feste Band-Reihenfolge (MUSS für beide Gebiete identisch sein!)
# band_order = ["B02", "B03", "B04", "B8A", "B12"]

# s2_bands = Path(r"C:\Users\Basti\Documents\Projekt_Waldbrand\wald\post_utm\resampled")

# bands = []
# for b in band_order:
#     band_path = next(s2_bands.glob(f"*{b}*.tiff"))
#     bands.append(read_band(band_path))

# bands = np.dstack(bands)
# print("Bänderformat Jasper:", bands.shape)

# # s2_bands = Path(r"C:\Users\felix\Documents\wald\post_utm\resampled")

# # bands = []
# # for band in s2_bands.glob("*.tiff"):
# #     data = read_band(band)
# #     bands.append(data)

# # # print(bands)
# # bands = np.dstack(bands)
# # print("Bänderformat:", bands.shape)

# ###### STACK BANDS Jasper ENDE ######


# ####### STACK BANDS Vinschgau ######

# # helper function for reading bands

# s2_bands_vinschgau = Path(r"C:\Users\Basti\Documents\Projekt_Waldbrand\wald\portugal_vila_real\resample")

# bands_vinschgau = []
# for b in band_order:
#     band_path = next(s2_bands_vinschgau.glob(f"*{b}*.tiff"))
#     bands_vinschgau.append(read_band(band_path))

# bands_vinschgau = np.dstack(bands_vinschgau)
# print("Bänderformat Vinschgau:", bands_vinschgau.shape)

# # s2_bands_vinschgau = Path(r"C:\Users\felix\Documents\wald\vinschgau\resampled")

# # bands_vinschgau = []
# # for band in s2_bands_vinschgau.glob("*.tiff"):
# #     data = read_band(band)
# #     bands_vinschgau.append(data)

# # # print(bands)
# # bands_vinschgau = np.dstack(bands_vinschgau)
# # print("Bänderformat Vinchgau:", bands.shape)

# ####### STACK BANDS Vinschgau ENDE ######


# ###### LABEL BLOCK ######

# y = read_band(r"C:\Users\Basti\Documents\Projekt_Waldbrand\wald\output_data\labels_burned_unburned.tiff")
# print("Labelformat", y.shape)

# ####### LABEL BLOCK ENDE ######


# ####### Reshaping Jasper ######

# rows, cols, n_bands = bands.shape

# X = bands.reshape((rows * cols, n_bands))
# y = y.reshape((rows * cols,))

# print("X shape after reshape:", X.shape)
# print("y shape after reshape:", y.shape)

# ####### Reshaping ENDE ######


# ####### Reshaping Vinschgau ######

# rows_v, cols_v, n_bands_v = bands_vinschgau.shape

# X_vinschgau = bands_vinschgau.reshape((rows_v * cols_v, n_bands_v))

# print("X_Vinschgau shape after reshape:", X_vinschgau.shape)

# ####### Reshaping ENDE ######


# ####### No-Data bearbeiten ######

# # eliminate no-data pixels from both S2 array and labels (no data value is -1)
# y_clean = y[y >= 0]
# X_clean = X[y >= 0, :]

# # check that the shape of the cleaned data sets matches
# print(X_clean.shape)
# print(y_clean.shape)

# ####### No-Data bearbeiten ENDE ######


# ####### Split Daten für ML und RF ######

# # extract train and test data and labels
# X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=RANDOM_STATE)

# n_trees = 300
# rf = RF(n_estimators=n_trees, n_jobs=-1, oob_score=True, random_state=RANDOM_STATE)

# rf.fit(X_train, y_train)

# # apply model on unseen test set
# y_pred = rf.predict(X_test)

# # calculate confusion matrix
# cnf_mat = confusion_matrix(y_test, y_pred)
# print(cnf_mat)
# print("OOB score:", rf.oob_score_)

# # plot confusion matrix
# labels = ["unburned", "burned"]

# cnf_mat = confusion_matrix(y_test, y_pred, labels=[0, 1])

# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cnf_mat,
#     display_labels=labels
# )

# disp.plot()
# plt.show()

# ####### Split Daten für ML und RF ENDE ######


# ####### Predict on full image (Jasper) and create GEO Output ######

# y_full_pred = rf.predict(X)

# # reshape to 2D array
# y_pred_all_2d = y_full_pred.reshape(rows, cols)

# # y_pred_all_2d[y >= 0] = y_full_pred # no Data 

# # read metadata of ONE BAND raster for output (damit es 1:1 zum ganzen Bild passt)
# template = {}
# template_raster = r"C:\Users\Basti\Documents\Projekt_Waldbrand\wald\post_utm\2024-09-05-00_00_2024-09-05-23_59_Sentinel-2_L2A_B02_(Raw).tiff"

# # herausschreiben auf ein vorhergenommenes raster
# with rasterio.open(template_raster, "r") as img:
#     template["crs"] = img.crs
#     template["transform"] = img.transform
#     template["height"] = img.height
#     template["width"] = img.width

# # write our predicted output as GeoTiff (ganzes Raster)
# with rasterio.open(
#     r"C:\Users\Basti\Documents\Projekt_Waldbrand\wald\output_data\predicted_labels_jasper_full_3.tif",
#     "w",
#     driver="GTiff",
#     crs=template["crs"],
#     transform=template["transform"],
#     width=template["width"],
#     height=template["height"],
#     count=1,
#     dtype=y_pred_all_2d.dtype,
# ) as fobj:
#     fobj.write(y_pred_all_2d, 1)

# print("fertig: predicted_labels_jasper_full.tif")

# ####### Predict on full image and create GEO Output for Jasper ENDE ######


# ####### Predict on full image (Vinchgau) and create GEO Output ######

# y_full_pred_Vinschgau = rf.predict(X_vinschgau)

# # reshape to 2D array
# y_pred_all_2d_Vinschgau = y_full_pred_Vinschgau.reshape(rows_v, cols_v)

# # y_pred_all_2d[y >= 0] = y_full_pred # no Data 

# # read metadata of ONE BAND raster for output (damit es 1:1 zum ganzen Bild passt)
# template = {}
# template_raster = r"C:\Users\Basti\Documents\Projekt_Waldbrand\wald\portugal_vila_real\resample\rs_2025-08-26-00_00_2025-08-26-23_59_Sentinel-2_L2A_B02_(Raw).tiff"

# # herausschreiben auf ein vorhergenommenes raster
# with rasterio.open(template_raster, "r") as img:
#     template["crs"] = img.crs
#     template["transform"] = img.transform
#     template["height"] = img.height
#     template["width"] = img.width

# # write our predicted output as GeoTiff (ganzes Raster)
# with rasterio.open(
#     r"C:\Users\Basti\Documents\Projekt_Waldbrand\wald\output_data\predicted_labels_portugal_full.tif",
#     "w",
#     driver="GTiff",
#     crs=template["crs"],
#     transform=template["transform"],
#     width=template["width"],
#     height=template["height"],
#     count=1,
#     dtype=y_pred_all_2d_Vinschgau.dtype,
# ) as fobj:
#     fobj.write(y_pred_all_2d_Vinschgau, 1)

# print("fertig: predicted_labels_vinschgau_full.tif")


# ####### Predict on full image and create GEO Output for Vinschgau ENDE ######

##############################################################
# Plot TRUE COLOR + CLASSIFICATION OVERLAY
##############################################################

ALPHA = 0.6

# Pfade

truecolor_dir = Path(
    r"C:\Users\Basti\Documents\Projekt_Waldbrand\wald\portugal_vila_real\resample"
)

prediction_path = Path(
    r"C:\Users\Basti\Documents\Projekt_Waldbrand\wald\output_data\predicted_labels_portugal_full.tif"
)

out_plot = Path(
    r"C:\Users\Basti\Documents\Projekt_Waldbrand\wald\output_data\overlay_truecolor_burned_portugal.png"
)


# Helper: visualization

def normalize(arr, pmin=2, pmax=98):
    arr = arr.astype(np.float32)
    vmin, vmax = np.percentile(arr[arr > 0], (pmin, pmax))
    arr = (arr - vmin) / (vmax - vmin)
    return np.clip(arr, 0, 1)

# Read True Color bands

b02 = rasterio.open(next(truecolor_dir.glob("*B02*.tif*"))).read(1)
b03 = rasterio.open(next(truecolor_dir.glob("*B03*.tif*"))).read(1)
b04 = rasterio.open(next(truecolor_dir.glob("*B04*.tif*"))).read(1)

# Read prediction

with rasterio.open(prediction_path) as src:
    pred = src.read(1)
    nodata = src.nodata

if nodata is None:
    nodata = 255

# Build RGB

rgb = np.dstack([
    normalize(b04),
    normalize(b03),
    normalize(b02)
])

# Overlay (nur burned = 1)

overlay = np.full(pred.shape, np.nan)
overlay[pred == 1] = 1

# Plot

plt.figure(figsize=(12, 12))
plt.imshow(rgb)
plt.imshow(overlay, cmap="YlOrRd", alpha=ALPHA)
plt.title("Burned Area – True Color Overlay")
plt.axis("off")
plt.tight_layout()
plt.savefig(out_plot, dpi=300)
plt.close()

print("Overlay gespeichert:")
print(out_plot)