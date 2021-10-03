# Load trained model and trained on random samples

from utils import config
from tensorflow.keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from utils import customLoss

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputDir", required=True, help="path to input directory of sample images")
args = vars(ap.parse_args())

print("[INFO] Loading images from inputDir")
testImagePaths = list(list_images(args["inputDir"]))
np.random.seed(42)
# randomly select 10 pairs from imagePaths
pairs = np.random.choice(testImagePaths, size=(10, 2))

model = load_model(config.CONTRASTIVE_MODEL_PATH_,  compile=False)

for (idx, (path1, path2)) in enumerate(pairs):
    # convert to gray scale image
    image1 = cv2.imread(path1, 0)
    image2 = cv2.imread(path2, 0)

    orig1 = image1.copy()
    orig2 = image2.copy()

    # add Channel dimension
    image1 = np.expand_dims(image1, axis=-1)
    image2 = np.expand_dims(image2, axis=-1)
    # add batch dimensions

    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)
    # rescale image between 0 to 1

    image1 = image1/255.0
    image2 = image2/255.0

    predictions = model.predict([image1,image2])
    prob = predictions[0][0]

    # initialize the figure
    fig = plt.figure("Image Pair #{}".format(idx + 1), figsize=(4, 2))
    plt.suptitle("Similarity: {:.2f}".format(prob))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(orig1, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(orig2, cmap=plt.cm.gray)
    plt.axis("off")

    # show the plot
    fname = 'output/similarity/contrastive_result' + str(idx)
    plt.savefig(fname)

