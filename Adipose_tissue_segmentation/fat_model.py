# Import needed packages

import tensorflow as tf

# Set tensorflow logging level
tf.logging.set_verbosity(tf.logging.INFO)


# Define the CNN U-Net model
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # regularizer l2
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    # batch normalization
    in_training = mode == tf.estimator.ModeKeys.TRAIN
    # dropout param
    keep_prob = 0.2

    # Input Layer
    input_layer = features["x"]

    # Convolutional Layer #1a and b
    conv1a = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv1a = tf.layers.batch_normalization(inputs=conv1a, training=in_training)
    conv1a = tf.nn.relu(conv1a)
    conv1a = tf.layers.dropout(inputs=conv1a, rate=keep_prob, training=in_training)

    conv1b = tf.layers.conv2d(
        inputs=conv1a,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv1b = tf.layers.batch_normalization(inputs=conv1b, training=in_training)
    conv1b = tf.nn.relu(conv1b)
    conv1b = tf.layers.dropout(inputs=conv1b, rate=keep_prob, training=in_training)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1b, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2a = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv2a = tf.layers.batch_normalization(inputs=conv2a, training=in_training)
    conv2a = tf.nn.relu(conv2a)
    conv2a = tf.layers.dropout(inputs=conv2a, rate=keep_prob, training=in_training)

    conv2b = tf.layers.conv2d(
        inputs=conv2a,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv2b = tf.layers.batch_normalization(inputs=conv2b, training=in_training)
    conv2b = tf.nn.relu(conv2b)
    conv2b = tf.layers.dropout(inputs=conv2b, rate=keep_prob, training=in_training)
    pool2 = tf.layers.max_pooling2d(inputs=conv2b, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3a = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv3a = tf.layers.batch_normalization(inputs=conv3a, training=in_training)
    conv3a = tf.nn.relu(conv3a)
    conv3a = tf.layers.dropout(inputs=conv3a, rate=keep_prob, training=in_training)
    conv3b = tf.layers.conv2d(
        inputs=conv3a,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv3b = tf.layers.batch_normalization(inputs=conv3b, training=in_training)
    conv3b = tf.nn.relu(conv3b)
    conv3b = tf.layers.dropout(inputs=conv3b, rate=keep_prob, training=in_training)
    pool3 = tf.layers.max_pooling2d(inputs=conv3b, pool_size=[2, 2], strides=2)

    # Convolutional Layer #4 and Pooling Layer #4
    conv4a = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv4a = tf.layers.batch_normalization(inputs=conv4a, training=in_training)
    conv4a = tf.nn.relu(conv4a)
    conv4a = tf.layers.dropout(inputs=conv4a, rate=keep_prob, training=in_training)
    conv4b = tf.layers.conv2d(
        inputs=conv4a,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv4b = tf.layers.batch_normalization(inputs=conv4b, training=in_training)
    conv4b = tf.nn.relu(conv4b)
    conv4b = tf.layers.dropout(inputs=conv4b, rate=keep_prob, training=in_training)
    pool4 = tf.layers.max_pooling2d(inputs=conv4b, pool_size=[2, 2], strides=2)

    # Middle convolution layers (bottom of U figure)
    conv5a = tf.layers.conv2d(
        inputs=pool4,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv5a = tf.layers.batch_normalization(inputs=conv5a, training=in_training)
    conv5a = tf.nn.relu(conv5a)
    conv5a = tf.layers.dropout(inputs=conv5a, rate=keep_prob, training=in_training)

    conv5b = tf.layers.conv2d(
        inputs=conv5a,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv5b = tf.layers.batch_normalization(inputs=conv5b, training=in_training)
    conv5b = tf.nn.relu(conv5b)
    conv5b = tf.layers.dropout(inputs=conv5b, rate=keep_prob, training=in_training)

    # Up-convolution 2d
    up_conv1 = tf.layers.conv2d_transpose(
        inputs=conv5b,
        filters=128,
        kernel_size=[3, 3],
        strides=(2, 2),
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    up_conv1 = tf.layers.batch_normalization(inputs=up_conv1, training=in_training)
    up_conv1 = tf.nn.relu(up_conv1)
    up_conv1 = tf.layers.dropout(inputs=up_conv1, rate=keep_prob, training=in_training)

    # Concatenate encoder and decoder info from same layer
    concat1 = tf.concat([conv4b, up_conv1], 3)

    # Convolutional Layer #6 and up-conv #2
    conv6a = tf.layers.conv2d(
        inputs=concat1,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv6a = tf.layers.batch_normalization(inputs=conv6a, training=in_training)
    conv6a = tf.nn.relu(conv6a)
    conv6a = tf.layers.dropout(inputs=conv6a, rate=keep_prob, training=in_training)

    conv6b = tf.layers.conv2d(
        inputs=conv6a,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv6b = tf.layers.batch_normalization(inputs=conv6b, training=in_training)
    conv6b = tf.nn.relu(conv6b)
    conv6b = tf.layers.dropout(inputs=conv6b, rate=keep_prob, training=in_training)

    up_conv2 = tf.layers.conv2d_transpose(
        inputs=conv6b,
        filters=64,
        kernel_size=[3, 3],
        strides=(2, 2),
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    up_conv2 = tf.layers.batch_normalization(inputs=up_conv2, training=in_training)
    up_conv2 = tf.nn.relu(up_conv2)
    up_conv2 = tf.layers.dropout(inputs=up_conv2, rate=keep_prob, training=in_training)

    concat2 = tf.concat([conv3b, up_conv2], 3)

    # Convolutional Layer #7 and up-conv #3
    conv7a = tf.layers.conv2d(
        inputs=concat2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv7a = tf.layers.batch_normalization(inputs=conv7a, training=in_training)
    conv7a = tf.nn.relu(conv7a)
    conv7a = tf.layers.dropout(inputs=conv7a, rate=keep_prob, training=in_training)

    conv7b = tf.layers.conv2d(
        inputs=conv7a,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv7b = tf.layers.batch_normalization(inputs=conv7b, training=in_training)
    conv7b = tf.nn.relu(conv7b)
    conv7b = tf.layers.dropout(inputs=conv7b, rate=keep_prob, training=in_training)

    up_conv3 = tf.layers.conv2d_transpose(
        inputs=conv7b,
        filters=32,
        kernel_size=[3, 3],
        strides=(2, 2),
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    up_conv3 = tf.layers.batch_normalization(inputs=up_conv3, training=in_training)
    up_conv3 = tf.nn.relu(up_conv3)
    up_conv3 = tf.layers.dropout(inputs=up_conv3, rate=keep_prob, training=in_training)

    concat3 = tf.concat([conv2b, up_conv3], 3)

    # Convolutional Layer #8 and up-conv #4
    conv8a = tf.layers.conv2d(
        inputs=concat3,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv8a = tf.layers.batch_normalization(inputs=conv8a, training=in_training)
    conv8a = tf.nn.relu(conv8a)
    conv8a = tf.layers.dropout(inputs=conv8a, rate=keep_prob, training=in_training)

    conv8b = tf.layers.conv2d(
        inputs=conv8a,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv8b = tf.layers.batch_normalization(inputs=conv8b, training=in_training)
    conv8b = tf.nn.relu(conv8b)
    conv8b = tf.layers.dropout(inputs=conv8b, rate=keep_prob, training=in_training)

    up_conv4 = tf.layers.conv2d_transpose(
        inputs=conv8b,
        filters=16,
        kernel_size=[3, 3],
        strides=(2, 2),
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    up_conv4 = tf.layers.batch_normalization(inputs=up_conv4, training=in_training)
    up_conv4 = tf.nn.relu(up_conv4)
    up_conv4 = tf.layers.dropout(inputs=up_conv4, rate=keep_prob, training=in_training)

    concat4 = tf.concat([conv1b, up_conv4], 3)

    # Convolutional Layer #9
    conv9a = tf.layers.conv2d(
        inputs=concat4,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv9a = tf.layers.batch_normalization(inputs=conv9a, training=in_training)
    conv9a = tf.nn.relu(conv9a)
    conv9a = tf.layers.dropout(inputs=conv9a, rate=keep_prob, training=in_training)

    conv9b = tf.layers.conv2d(
        inputs=conv9a,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    conv9b = tf.layers.batch_normalization(inputs=conv9b, training=in_training)
    conv9b = tf.nn.relu(conv9b)
    conv9b = tf.layers.dropout(inputs=conv9b, rate=keep_prob, training=in_training)

    # Prediction layer conv 1x1
    pixel_pred = tf.layers.conv2d(
        inputs=conv9b,
        filters=1,
        kernel_size=[1, 1],
        padding="same",
        activation=None,
        kernel_regularizer=regularizer,
    )
    pixel_pred = tf.layers.batch_normalization(inputs=pixel_pred, training=in_training)
    pixel_pred = tf.nn.sigmoid(pixel_pred)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": pixel_pred,
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.identity(pixel_pred, name="pred"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions["classes"])

    # Calculate Loss (for both TRAIN and EVAL modes)
    l2_loss = tf.losses.get_regularization_loss()
    loss = tf.losses.log_loss(labels=labels, predictions=pixel_pred) + 0.01 * l2_loss

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step()
            )
        tf.summary.scalar("loss", loss)
        summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir=SAVE_PATH + "/summary",
            summary_op=tf.summary.merge_all(),
        )
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_hook]
        )

    # Configure the Evaluation Op (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "mse": tf.metrics.mean_squared_error(labels=labels, predictions=pixel_pred)
        }
        tf.summary.scalar("mse_scalar", eval_metric_ops["mse"])
        summary_eval_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir=SAVE_PATH + "/eval",
            summary_op=tf.summary.merge_all(),
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            training_hooks=[summary_eval_hook],
        )
