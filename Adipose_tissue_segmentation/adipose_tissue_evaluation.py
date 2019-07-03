import fat_model as fat_code
import tensorflow as tf

# PATH to the model weights file
FAT_PATH = "...\\unet_fat_segmentation"

#Import the model and load the weights
fat_detector = tf.estimator.Estimator(model_fn=fat_code.cnn_model_fn, model_dir=FAT_PATH)

#Predict on an image
pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": IMAGE},
        batch_size=1,
        num_epochs=1,
        shuffle=False)
fat_pred = fat_detector.predict(input_fn=pred_input_fn)
predict_fat = []
for j in fat_pred:
    predict_fat.append(j)

#Threshold the prediction and apply morphological operations to extract the TDLUs
#Apply threshhold
thresh = 0.6
image = predict_fat > thresh
