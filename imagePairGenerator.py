
import numpy as np


def generatePairs(images, labels):
    pairImages = []
    pairLabels = []

    numClasses = len(np.unique(labels))
    print(numClasses)
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

    for idxA in range(len(images)):
        currentImage = images[idxA]
        currentLabel = labels[idxA]

        # get positive example for current Image with same class label

        idxPos = np.random.choice(idx[currentLabel])
        positiveImage = images[idxPos]

        # append positive image
        pairImages.append([currentImage, positiveImage])
        pairLabels.append([1])

        # get Negative Example for current Image with different class label
        negIdx = np.where(labels != currentLabel)[0]
        idxNeg = np.random.choice(negIdx)
        negativeImage = images[idxNeg]

        # add negative pair to pair images
        pairImages.append([currentImage, negativeImage])
        pairLabels.append([0])

    return pairImages, pairLabels

