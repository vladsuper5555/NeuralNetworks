import pip
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO

def install(package):
    pip.main(['install', package])

try:
    from torchvision.datasets import MNIST
    print("module 'torchvision' is installed")
except ModuleNotFoundError:
    print("module 'torchvision' is not installed")
    install("torchvision")

import numpy as np
from torchvision.datasets import MNIST
import time

def download_mnist(is_train: bool):
    return MNIST(root='./data', download=True, train=is_train)

train_dataset = download_mnist(True)
test_dataset = download_mnist(False)

def processData(dataset):
    images = np.array([np.array(img, dtype=np.float32).flatten() for img, _ in dataset]) / 255.0
    labels = np.array([label for _, label in dataset], dtype=np.uint8)
    return images, labels

np_train_dataset = processData(train_dataset)
np_test_dataset = processData(test_dataset)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)

def softmax(z):
    z_stable = z - np.max(z)
    exp_values = np.exp(z_stable)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Layer:
    def __init__(self, previousLayerNumberOfNeurons, numberOfNeurons, is_output=False):
        self.is_output = is_output
        # Xavier initialization
        limit = np.sqrt(6 / (previousLayerNumberOfNeurons + numberOfNeurons))
        self.weights = np.random.uniform(-limit, limit, (numberOfNeurons, previousLayerNumberOfNeurons))
        self.bias = np.zeros((numberOfNeurons, 1))
        self.activations = None
        self.z_values = None

    def forwardPropagation(self, inputs):
        self.inputs = inputs  # Save inputs for backward pass
        self.z_values = inputs @ self.weights.T + self.bias.T
        if self.is_output:
            self.activations = softmax(self.z_values)
        else:
            self.activations = sigmoid(self.z_values)
        return self.activations

    def backwardPropagation(self, error):
        if self.is_output:
            dz = error  # For softmax with cross-entropy loss
        else:
            dz = error * sigmoid_derivative(self.activations)
        return dz

input_size = 784  
layer_sizes = [100]
output_size = 10

layers = []
previous_layer_size = input_size

for size in layer_sizes:
    layers.append(Layer(previous_layer_size, size))
    previous_layer_size = size

layers.append(Layer(previous_layer_size, output_size, is_output=True))

LEARNING_RATE = 0.09
EPOCH_NUMBER = 50
BATCH_SIZE = 100
DECAY_FACTOR = 0.5
PATIENCE = 5
MIN_DELTA = 0.001
MIN_LR = 1e-5 

best_train_accuracy = 0
epochs_without_improvement = 0


def runBatch(np_batch_train_dataset):
    batch_size = np_batch_train_dataset[0].shape[0]
    num_classes = output_size

    activations = np_batch_train_dataset[0]
    for layer in layers:
        activations = layer.forwardPropagation(activations)

    softmax_output = activations

    targetArray = np.zeros((batch_size, num_classes))
    targetArray[np.arange(batch_size), np_batch_train_dataset[1]] = 1

    errorArray = softmax_output - targetArray

    gradients_accumulated = [np.zeros_like(layer.weights) for layer in layers]
    bias_accumulated = [np.zeros_like(layer.bias) for layer in layers]

    dz = errorArray

    for i in reversed(range(len(layers))):
        layer = layers[i]
        dz = layer.backwardPropagation(dz)
        inputs = layer.inputs
        gradients_accumulated[i] = dz.T @ inputs / batch_size
        bias_accumulated[i] = np.mean(dz, axis=0, keepdims=True).T
        if i > 0:
            dz = dz @ layer.weights  # backpropagate error

    return gradients_accumulated, bias_accumulated

def runEpoch():
    total_samples = np_train_dataset[0].shape[0]
    indices = np.arange(total_samples)
    # np.random.shuffle(indices)
    shuffled_images = np_train_dataset[0][indices]
    shuffled_labels = np_train_dataset[1][indices]

    batch_count = total_samples // BATCH_SIZE

    for batchIndex in range(batch_count):
        batch_start = batchIndex * BATCH_SIZE
        batch_end = (batchIndex + 1) * BATCH_SIZE
        batch_train_dataset = [shuffled_images[batch_start:batch_end],
                               shuffled_labels[batch_start:batch_end]]

        # Run forward and backward pass for batch
        gradients_accumulated, bias_accumulated = runBatch(batch_train_dataset)

        # Update weights and biases
        for i, layer in enumerate(layers):
            layer.weights -= LEARNING_RATE * gradients_accumulated[i]
            layer.bias -= LEARNING_RATE * bias_accumulated[i]

def main():
    global LEARNING_RATE, best_train_accuracy, epochs_without_improvement
    for epochIndex in range(EPOCH_NUMBER):
        start_time = time.time()
        runEpoch()
        epoch_duration = time.time() - start_time
        print(f"Epoch {epochIndex + 1} training time: {epoch_duration:.2f} seconds")

        train_activations = np_train_dataset[0]
        correct_train_labels = np_train_dataset[1]
        correctlyPredicted_train = 0

        for layer in layers:
            train_activations = layer.forwardPropagation(train_activations)

        train_predictions = np.argmax(train_activations, axis=1)
        correctlyPredicted_train += np.sum(train_predictions == correct_train_labels)

        train_accuracy = correctlyPredicted_train / np_train_dataset[0].shape[0]
        train_error = 1 - train_accuracy
        print(f"Training accuracy at epoch {epochIndex + 1}: {train_accuracy:.4f}")

        if train_accuracy > best_train_accuracy + MIN_DELTA:
            best_train_accuracy = train_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            if LEARNING_RATE > MIN_LR:
                LEARNING_RATE *= DECAY_FACTOR
                LEARNING_RATE = max(LEARNING_RATE, MIN_LR)  # Ensure LR does not go below MIN_LR
                print(f"Learning rate decayed to {LEARNING_RATE:.6f}")
            epochs_without_improvement = 0  # Reset counter after adjusting learning rate




        # Testing
        start_time = time.time()
        tests = np_test_dataset[0]
        correctLabels = np_test_dataset[1]
        correctlyPredicted = 0

        activations = tests
        for layer in layers:
            activations = layer.forwardPropagation(activations)

        predictions = np.argmax(activations, axis=1)
        correctlyPredicted += np.sum(predictions == correctLabels)

        test_duration = time.time() - start_time
        test_accuracy = correctlyPredicted / tests.shape[0]
        print(f"Accuracy on tests at epoch {epochIndex + 1}: {test_accuracy:.4f}")
        print(f"Testing time: {test_duration:.2f} seconds\n")

class DrawingApp:
    def __init__(self, root, model_layers):
        self.root = root
        self.model_layers = model_layers
        self.root.title("Draw a Digit")
        self.root.geometry("280x330")
        self.canvas = tk.Canvas(self.root, bg="black", width=280, height=280)
        self.canvas.pack()

        # Buttons for actions
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()
        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=0, column=0, padx=5, pady=5)
        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=0, column=1, padx=5, pady=5)

        self.canvas.bind("<B1-Motion>", self.paint)

        # To store drawn paths
        self.image = Image.new("L", (280, 280), 0)  # 'L' mode for grayscale
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        thickness = 6
        if not hasattr(self, 'last_x') or self.last_x == None:
            self.last_x, self.last_y = event.x, event.y

        # Draw a line between the last point and current position
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill="white", width=thickness)
        self.draw.line([self.last_x, self.last_y, event.x, event.y], fill=255, width=thickness)

        # Update last point to the current one
        self.last_x, self.last_y = event.x, event.y
        def reset_last_x(event):
            self.last_x = None

        self.canvas.bind("<ButtonRelease-1>", reset_last_x)


    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=0)
        self.last_x = None

    def predict_digit(self):
        # Resize to 28x28 and flatten the image
        resized_image = self.image.resize((28, 28), Image.LANCZOS)
        image_data = np.array(resized_image, dtype=np.float32).reshape(784) / 255.0
        resized_image.show()
        # Pass through the model
        activations = image_data
        for layer in self.model_layers:
            activations = layer.forwardPropagation(activations)

        # Get prediction
        prediction = np.argmax(activations, axis=1)
        print(f"Predicted Digit: {prediction[0]} Confidence: {activations[0][prediction[0]]} \n All confidence: {activations[0]}")

if __name__ == "__main__":
    main()
    root = tk.Tk()
    app = DrawingApp(root, layers)  # 'layers' is from your existing neural network model
    root.mainloop()