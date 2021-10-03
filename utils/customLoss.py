import tensorflow.keras.backend as KB
import tensorflow as tf


def contrastiveloss(y_true, y_pred, margin=1):
    # explicit cast y_true to y_pred to avoid tensorflow errors
    y_true = tf.cast(y_true, y_pred.dtype)

    # contrastive loss = y_true * ||DW||_2 + (1-y_true) * ( max(margin - ||DW||_2 , 0))
    # ||DW||_2
    squared_predictions = KB.square(y_pred)

    squared_margin = KB.square(KB.maximum(margin - y_pred, 0))

    # return avg loss over all training batch
    return KB.mean(y_true * squared_predictions + (1 - y_true) * squared_margin)
