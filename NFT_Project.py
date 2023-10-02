import numpy as np
import cv2
import os
import random

# Function to compute HOG features
def compute_hog_features(image):
    win_size = (50, 50)
    block_size = (10, 10)
    block_stride = (5, 5)
    cell_size = (5, 5)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    return hog.compute(image).flatten()

# Activation function and its derivative
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights
def initialize_weights(input_size, hidden_size, output_size):
    w1 = np.random.randn(input_size, hidden_size)
    w2 = np.random.randn(hidden_size, output_size)
    return w1, w2

# Read and preprocess images
folders = ['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn']
dataset = []
labels = []

for i, folder in enumerate(folders):
    folder_path = f'./dataset/{folder}'
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized_img = cv2.resize(img, (50, 50))
            hog_features = compute_hog_features(resized_img)
            dataset.append(hog_features)
            labels.append(i)

# Update input size based on HOG feature length
input_size = len(dataset[0])

# Initialize weights
hidden_size = 128
output_size = 6  # Number of classes (King, Queen, Rook, Bishop, Knight, Pawn)
w1, w2 = initialize_weights(input_size, hidden_size, output_size)

# Randomize dataset
combined = list(zip(dataset, labels))
random.shuffle(combined)
dataset, labels = zip(*combined)

# Convert to numpy arrays
dataset = np.array(dataset)
labels = np.array(labels)

# Training parameters
epochs = 10
alpha = 0.1  # Learning rate

# Initialize a small regularization term
lambda_reg = 0.001
decay_factor = 0.95

# Training loop with early stopping based on a validation set
best_val_accuracy = 0
patience = 3
wait = 0

# Split dataset into training and validation sets
val_split = 0.1
val_count = int(val_split * len(dataset))
train_dataset = dataset[val_count:]
train_labels = labels[val_count:]
val_dataset = dataset[:val_count]
val_labels = labels[:val_count]

# Training loop
for epoch in range(epochs):
    # Shuffle the training data for each epoch
    combined = list(zip(train_dataset, train_labels))
    random.shuffle(combined)
    train_dataset, train_labels = zip(*combined)
    
    for i in range(len(dataset)):
        # Step 1: Feedforward
        x = dataset[i]
        z_in = np.dot(x, w1)
        z = sigmoid(z_in)
        y_in = np.dot(z, w2)
        y = sigmoid(y_in)

        # Step 6: Backpropagation of error
        target = np.zeros(output_size)
        target[labels[i]] = 1
        delta_y = (target - y) * sigmoid_derivative(y)
        delta_z = np.dot(delta_y, w2.T) * sigmoid_derivative(z)

        # Update weights and biases (Step 8)
        w2 += alpha * np.outer(z, delta_y)
        w1 += alpha * np.outer(x, delta_z)
        
        # Add regularization to weight updates
        w2 += -lambda_reg * w2
        w1 += -lambda_reg * w1
        
    # Validate the model on the validation set
    correct_val = 0
    for i in range(len(val_dataset)):
        x = val_dataset[i]
        z_in = np.dot(x, w1)
        z = sigmoid(z_in)
        y_in = np.dot(z, w2)
        y = sigmoid(y_in)

        if np.argmax(y) == val_labels[i]:
            correct_val += 1

    val_accuracy = correct_val / len(val_dataset)
    print(f"Validation Accuracy at epoch {epoch}: {val_accuracy * 100}%")

    # Early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        wait = 0
    else:
        wait += 1

    if wait >= patience:
        print("Early stopping due to no improvement in validation accuracy.")
        break     
    
    # Reduce the learning rate
    alpha *= decay_factor

# Read and preprocess test images
test_folders = ['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn']
test_dataset = []
test_labels = []

for i, folder in enumerate(test_folders):
    folder_path = f'./test_dataset/{folder}'
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized_img = cv2.resize(img, (50, 50))
            hog_features = compute_hog_features(resized_img)
            test_dataset.append(hog_features)
            test_labels.append(i)

# Convert to numpy arrays for testing
test_dataset = np.array(test_dataset)
test_labels = np.array(test_labels)

# Test the model
correct = 0
for i in range(len(test_dataset)):
    x = test_dataset[i]
    z_in = np.dot(x, w1)
    z = sigmoid(z_in)
    y_in = np.dot(z, w2)
    y = sigmoid(y_in)

    if np.argmax(y) == test_labels[i]:
        correct += 1

test_accuracy = correct / len(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100}%")
