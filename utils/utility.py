import tensorflow.keras.backend as KB
import matplotlib.pyplot as plt


def euclideanDistance(tensors):

    tensorA, tensorB = tensors

    squaredErr = KB.sum(KB.square(tensorA-tensorB), axis=1, keepdims=True)

    return KB.sqrt(KB.maximum(squaredErr, KB.epsilon()))


def plot_model(model_history, PLOT_PATH):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(model_history.history["loss"], label="train_loss")
    plt.plot(model_history.history["val_loss"], label="val_loss")
    plt.plot(model_history.history["accuracy"], label="train_acc")
    plt.plot(model_history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)