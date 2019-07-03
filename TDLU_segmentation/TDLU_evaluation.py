import tdlu_model as tdlu_code
import tensorflow as tf
import cv2
from skimage import morphology

# PATH to the model weights file
TDLU_PATH = "...\\unet_newcases_revised"

#Import the model and load the weights
tdlu_detector = tf.estimator.Estimator(model_fn=tdlu_code.cnn_model_fn, model_dir=TDLU_PATH)

#Predict on an image
pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": IMAGE},
        batch_size=1,
        num_epochs=1,
        shuffle=False)
tdlu_pred = tdlu_detector.predict(input_fn=pred_input_fn)
predict_tdlu = []
for j in tdlu_pred:
    predict_tdlu.append(j)

#Threshold the prediction and apply morphological operations to extract the TDLUs
#Apply threshhold
thresh = 60
image = predict_tdlu > thresh
#Remove small objects (noise)
image = morphology.remove_small_objects(image.astype(bool), 20000)
#Filter the image
image = image.astype('uint8')
image = cv2.medianBlur(image,11)
#Remove small objects & fill small holes
image = morphology.remove_small_objects(image.astype(bool), 20000)
image = morphology.remove_small_holes(image.astype(bool), 1000000)