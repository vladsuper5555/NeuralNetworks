import pip
def install(package):
    pip.main(['install', package])

try:
    from torchvision.datasets import MNIST
    print("module 'mutagen' is installed")
except ModuleNotFoundError:
    print("module 'torchvsion' is not installed")
    # or
    install("torchvision") # the install function from the question

import numpy as np
from torchvision.datasets import MNIST
train_dataset = None
test_dataset = None
def download_mnist(is_train: bool):
    if is_train == True:
        return MNIST(root='./data', transform=lambda x: np.array(x).flatten(), download=True, train=is_train)
    else: 
        return MNIST(root='./data', transform=lambda x: np.array(x).flatten(), download=True, train=is_train)

train_dataset = download_mnist(True)
test_dataset = download_mnist(False)

def processData(dataset):
    np_dataset_images = np.array([object[0] for object in dataset], dtype = np.bool_).reshape(len(dataset), 28 * 28)
    np_dataset_labels = np.array([object[1] for object in dataset], dtype = np.uint8)
    
    return np_dataset_images, np_dataset_labels

np_train_dataset = processData(train_dataset)
np_test_dataset = processData(test_dataset)


import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    z_stable = z - np.max(z)  # for numerical stability
    exp_values = np.exp(z_stable)
    return exp_values / np.sum(exp_values)

def cross_entropy_loss(predicted, target):
    # Use small epsilon to prevent log(0)
    epsilon = 1e-12
    predicted = np.clip(predicted, epsilon, 1 - epsilon)
    return -np.sum(target * np.log(predicted))

class Layer:
    def __init__(self, previousLayerNumberOfNeurons, numberOfNeurons, is_output=False):
        self.is_output = is_output
        # Xavier initialization
        limit = np.sqrt(6 / (previousLayerNumberOfNeurons + numberOfNeurons))
        self.weights = np.random.uniform(-limit, limit, (numberOfNeurons, previousLayerNumberOfNeurons))
        self.bias = np.zeros(numberOfNeurons)
        self.activations = None
        self.z_values = None

    def forwardPropagation(self, inputs):
        self.z_values = (self.weights @ inputs) + self.bias
        
        # Apply the activation function
        if self.is_output:
            self.activations = softmax(self.z_values)
        else:
            self.activations = sigmoid(self.z_values)
        return self.activations

    def backwardPropagation(self, error):
        if self.is_output:
            dz = error
        else:
            dz = error * sigmoid_derivative(self.z_values)

        return dz

# Define network architecture
input_size = 784  # Number of input features
layer_sizes = [100]  # Sizes of each layer
output_size = 10  # Number of output classes

# Initialize layers as a list of Layer objects
layers = []

# Input layer to first hidden layer
layers.append(Layer(input_size, layer_sizes[0]))

# Hidden layers
for i in range(0, len(layer_sizes)):
    layers.append(Layer(layer_sizes[i - 1], layer_sizes[i]))

# Output layer
layers.append(Layer(layer_sizes[-1], output_size, is_output=True))



LEARNING_RATE = 0.01
EPOCH_NUMBER = 100


from concurrent.futures import ThreadPoolExecutor
# def runBatch(np_batch_train_dataset):
#     batch_size = np_batch_train_dataset[0].shape[0]
#     num_classes = output_size

#     # Initialize accumulators for gradients and biases
#     gradients_accumulated = [np.zeros_like(layer.weights) for layer in layers]
#     bias_accumulated = [np.zeros_like(layer.bias) for layer in layers]
#     batch_correct = 0

#     # Function to compute forward pass and error for a single sample
#     def process_sample(testIndex):
#         activations = np_batch_train_dataset[0][testIndex]
#         for perceptron in layers:
#             activations = perceptron.forwardPropagation(activations)
        
#         softMaxArray = activations
#         correctPredictionValue = np_batch_train_dataset[1][testIndex]
#         targetArray = np.zeros(num_classes)
#         targetArray[correctPredictionValue] = 1
#         is_correct = int(correctPredictionValue == np.argmax(softMaxArray))
        
#         # Return error for backpropagation
#         errorArray = softMaxArray - targetArray
#         return errorArray, is_correct

#     # Run forward pass in parallel and gather errors and results
#     with ThreadPoolExecutor() as executor:
#         results = list(executor.map(process_sample, range(batch_size)))

#     # Accumulate errors for each sample and update batch loss and correct count
#     for errorArray, is_correct in results:
#         batch_correct += is_correct

#         # Backpropagation: accumulate gradients and biases for each layer
#         for i in reversed(range(len(layers))):
#             if i == 0:
#                 continue
#             layer = layers[i]
#             dz = layer.backwardPropagation(errorArray)
#             gradients_accumulated[i] += np.outer(dz, layers[i - 1].activations)

#             bias_accumulated[i] += dz
#             errorArray = layer.weights.T @ dz

#     # Average the accumulated gradients and biases over the batch size
#     # gradients_accumulated = [grad / batch_size for grad in gradients_accumulated]
#     # bias_accumulated = [bias / batch_size for bias in bias_accumulated]

#     return gradients_accumulated, bias_accumulated, batch_correct


def runBatch(np_batch_train_dataset):
    batch_size = np_batch_train_dataset[0].shape[0]
    num_classes = output_size

    # Initialize accumulators for gradients and biases
    gradients_accumulated = [np.zeros_like(layer.weights) for layer in layers]
    bias_accumulated = [np.zeros_like(layer.bias) for layer in layers]
    batch_correct = 0

    # Sequentially process each sample in the batch
    for testIndex in range(batch_size):
        activations = np_batch_train_dataset[0][testIndex]
        for perceptron in layers:
            activations = perceptron.forwardPropagation(activations)
        
        softMaxArray = activations
        correctPredictionValue = np_batch_train_dataset[1][testIndex]
        targetArray = np.zeros(num_classes)
        targetArray[correctPredictionValue] = 1
        is_correct = int(correctPredictionValue == np.argmax(softMaxArray))
        
        # Update batch correct count
        batch_correct += is_correct

        # Calculate error for backpropagation
        errorArray = softMaxArray - targetArray

        # Accumulate gradients and biases for each layer
        for i in reversed(range(len(layers))):
            if i == 0:
                continue
            layer = layers[i]
            dz = layer.backwardPropagation(errorArray)
            gradients_accumulated[i] += np.outer(dz, layers[i - 1].activations)
            bias_accumulated[i] += dz
            errorArray = layer.weights.T @ dz

    # Return accumulated gradients and biases, and count of correct predictions
    return gradients_accumulated, bias_accumulated, batch_correct



def runEpoch(batch_size=100):
    total_samples = np_train_dataset[0].shape[0]
    batch_count = total_samples // batch_size
    total_loss = 0
    total_correct = 0

    for batchIndex in range(batch_count):
        batch_start = batchIndex * batch_size
        batch_end = (batchIndex + 1) * batch_size
        batch_train_dataset = [np_train_dataset[0][batch_start:batch_end], 
                               np_train_dataset[1][batch_start:batch_end]]

        # Run forward and backward pass for batch
        gradients_accumulated, bias_accumulated, batch_correct = runBatch(batch_train_dataset)

        # Update weights and biases for each perceptron
        for i, layer in enumerate(layers):
            # print ("layer " + str(i))
            # print(" gradients: ")
            # print(gradients_accumulated[i])
            # print("weghts: ")
            # print(layers[i].weights)
            # time.sleep(2)
            layer.weights -= LEARNING_RATE * gradients_accumulated[i]
            layer.bias -= LEARNING_RATE * bias_accumulated[i]

        total_correct += batch_correct

    # avg_loss = total_loss / batch_count
    # accuracy = total_correct / total_samples

    # return avg_loss, accuracy
    return total_correct

def runTest(inputs):
    activations = inputs
    for i, perceptron in enumerate(layers):
        activations = perceptron.forwardPropagation(activations)
    max_index = np.argmax(activations)
    return max_index

def main ():
    for epochIndex in range(EPOCH_NUMBER):
        totalCorrect = runEpoch()
        print('trainingAccuracy = ', totalCorrect, (np_train_dataset[0].size // (28 * 28)), totalCorrect / (np_train_dataset[0].size // (28 * 28)))
        correctlyPredicted = 0
        tests = np_test_dataset[0]
        correctPredictions = np_test_dataset[1]
        for index in range(tests.size // (28 * 28)):
            prediction = runTest(tests[index])
            if prediction == correctPredictions[index]:
                correctlyPredicted += 1
        print('Accuracy on tests at epoch ' + str(epochIndex) + " : " + str(correctlyPredicted / (tests.size // (28 * 28))))
main()