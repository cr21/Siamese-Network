import os

BATCH_SIZE = 64


# input shape (h,w,c)
IMG_SHAPE = (28, 28, 1)

# number of epochs to train
EPOCHS = 10

# output base directory
BASE_OUTPUT = 'output'

# Model path to store trained model
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])

# plot path of training accuracy, training loss, validation accuracy, validation loss

PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot_binary.png"])
