from skimage.feature import peak_local_max
import acini_model as acini_code

# PATH to the model weights file
ACINI_PATH = "...\\unet_gaussian.hdf5"

#Import the model and load the weights
model = acini_code.unet()
model.load_weights(ACINI_PATH)

#Predict on an image
predict_acini = model.predict(IMAGE/255, batch_size=1)

#Use non-maximum suppression to find the acini coordinates
acini_coordinates = peak_local_max(predict_acini, min_distance=30, threshold_abs=0.48, exclude_border=False)