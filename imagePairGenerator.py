from tensorflow.keras.datasets import mnist
from imutils import build_montages
import numpy as np
import cv2


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


print("[INFO] loading MNIST DATASET")
(trainX, trainY), (testX, testY) = mnist.load_data()

print("[INFO] preparing Positive and Negative images")
(pairTrain, labelTrain) = generatePairs(trainX, trainY)
(pairTest, labelTest) = generatePairs(testX, testY)

images = []

for i in np.random.choice(np.arange(0, len(pairTrain)), size=49):
    imageA = pairTrain[i][0]
    imageB = pairTrain[i][1]
    label = labelTrain[i]

    output = np.zeros((36, 60), dtype="uint8")
    pair = np.hstack([imageA, imageB])
    output[4:32, 0:56] = pair

    # text assignment to image
    text = "neg" if label[0] == 0 else "pos"
    color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)

    vis = cv2.merge([output] * 3)
    vis = cv2.resize(vis, (96, 51), interpolation=cv2.INTER_LINEAR)
    cv2.putText(vis, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    images.append(vis)

montage = build_montages(images, (96, 51), (7, 7))[0]

cv2.imshow("SIMEASE IAMGE PAIR", montage)
cv2.waitKey(0)
cv2.imwrite('imagepair.png', montage)
