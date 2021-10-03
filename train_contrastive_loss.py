from models import simese_network
from utils import config
from utils import utility
from utils.customLoss import contrastiveloss
from utils.imagePairGenerator import generatePairs
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
from utils import customLoss
import numpy as np

print("[INFO] Loading Training Dataset")
((trainX, trainY), (testX, testY)) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0

# Add Channel dimension

trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
print(trainX.shape)

print("[INFO] preparing train positive and negative image pairs")
(trainPairs, trainLabels) = generatePairs(trainX, trainY)

print("[INFO] preparing test positive and negative image pairs")
(testPairs, testLabels) = generatePairs(testX, testY)

print("[INFO] preparing siamese network")

imageA = Input(config.IMG_SHAPE)
imageB = Input(config.IMG_SHAPE)

siamese_backbone = simese_network.buildSimese_backbone(config.IMG_SHAPE)
featureA = siamese_backbone(imageA)
featureB = siamese_backbone(imageB)

distance = Lambda(utility.euclideanDistance)([featureA, featureB])
outputs = Dense(1, activation='sigmoid')(distance)
model = Model(inputs=[imageA, imageB], outputs=outputs)

print("[INFO] compiling model")

model.compile(loss=customLoss.contrastiveloss, optimizer='adam', metrics=["accuracy"])
# compile the model


# train the model
print("[INFO] Start training Model")
model_history = model.fit(
    [np.array(trainPairs)[:, 0], np.array(trainPairs)[:, 1]], np.array(trainLabels)[:],
    validation_data=([np.array(testPairs)[:, 0], np.array(testPairs)[:, 1]], np.array(testLabels)[:]),
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS)

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.CONTRASTIVE_MODEL_PATH_)

# plot the training history
print("[INFO] plotting training history...")
utility.plot_model(model_history, config.CONTRASTIVE_PLOT_PATH)
