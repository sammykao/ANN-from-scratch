import numpy as np

class NeuralNetwork:
    def __init__(self, i_layer_size, h_layer_size, o_layer_size):
        self.i_layer_size = i_layer_size
        self.h_layer_size = h_layer_size
        self.o_layer_size = o_layer_size
        
        # Start with random weights and B values to the
        self.weight_input_hlayer = np.random.randn(self.i_layer_size, self.h_layer_size)
        self.b_input_hlayer = np.zeros((1, self.h_layer_size))
        self.weight_input_olayer = np.random.randn(self.h_layer_size, self.o_layer_size)
        self.bias_h_layer_output = np.zeros((1, self.o_layer_size))
        
    def forward(self, X):
        # Forward propagation
        self.h_layer_input = np.dot(X, self.weight_input_hlayer) + self.b_input_hlayer
        self.h_layer_output = self.sigmoid(self.h_layer_input)
        self.output = np.dot(self.h_layer_output, self.weight_input_olayer) + self.bias_h_layer_output
        self.output_probabilities = self.softmax(self.output)
        return self.output_probabilities
        
    def backward(self, X, y, learning_rate):
        # Backward propagation
        m = X.shape[0]
        # Compute gradients
        output_error = self.output_probabilities - y
        hidden_error = np.dot(output_error, self.weight_input_olayer.T) * self.sigmoid_derivative(self.h_layer_output)
        # Update weights and biases
        self.weight_input_olayer -= learning_rate * np.dot(self.h_layer_output.T, output_error) / m
        self.bias_h_layer_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True) / m
        self.weight_input_hlayer -= learning_rate * np.dot(X.T, hidden_error) / m
        self.b_input_hlayer -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True) / m
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    
    def train(self, X, y, X_val, y_val, epochs, learning_rate):
        best_val_loss = float('inf')
        no_improvement_count = 0

        for epoch in range(epochs):
            # Forward pass
            output_probabilities = self.forward(X)
            
            # One-hot encode the labels
            y_one_hot = np.eye(self.o_layer_size)[y]
            
            # Backward pass
            self.backward(X, y_one_hot, learning_rate)
            
            # Compute training loss
            train_loss = self.cross_entropy_loss(output_probabilities, y_one_hot)
            
            # Compute validation loss
            val_output_probabilities = self.forward(X_val)
            val_loss = self.cross_entropy_loss(val_output_probabilities, np.eye(self.o_layer_size)[y_val])
            
            # Compute training accuracy
            train_accuracy = self.accuracy(output_probabilities, y)
            
            # Compute validation accuracy
            val_accuracy = self.accuracy(val_output_probabilities, y_val)
            
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss: .5f}, ' 
            f'Training Accuracy: %{train_accuracy * 100:.5f}, '
            f' Validation Loss: {val_loss:.5f}, Validation Accuracy: %{val_accuracy * 100:.5f}')
        
            
            # Check for improvement in validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
                # If no improvement for 'patience' epochs, stop training
                if no_improvement_count >= 10:
                    print(f'No improvement in validation loss for 10 epochs. STOPPING TRAINING AT CURRENT WEIGHTS')
                    break


    def cross_entropy_loss(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 0, 1)
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def accuracy(self, output_probabilities, y):
        # Get predicted labels
        predicted_labels = np.argmax(output_probabilities, axis=1)
        # Compute accuracy
        accuracy = np.mean(predicted_labels == y)
        return accuracy