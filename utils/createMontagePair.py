from tensorflow.keras.datasets import mnist
from imutils import build_montages
import numpy as np
import cv2
from utils.imagePairGenerator import generatePairs

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

    output = np.zeros((72, 120), dtype="uint8")
    pair = np.hstack([imageA, imageB])
    pair = cv2.resize(pair, (112, 56))
    output[8:64, 0:112] = pair

    # text assignment to image
    text = "neg" if label[0] == 0 else "pos"
    color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)

    colorImage = cv2.merge([output] * 3)
    cv2.putText(colorImage, text, (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    images.append(colorImage)

montage = build_montages(images, (216, 120), (5, 5))[0]


cv2.imshow("SIAMESE IMAGE PAIR", montage)
cv2.waitKey(0)
cv2.imwrite('../output/imagePair.png', montage)
