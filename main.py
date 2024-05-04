from dataprocessor import preprocess_data
from neuralnetwork import NeuralNetwork
 
if __name__ == "__main__":
    x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data()
    
    # Example usage:
    input_size = 4
    hidden_size = 8
    output_size = 3
    learning_rate = 0.05
    epochs = 10000

    model = NeuralNetwork(input_size, hidden_size, output_size)

    # Train the model
    model.train(x_train, y_train, x_val, y_val, epochs=epochs, learning_rate=learning_rate)

    # Evaluate the model
    predictions = model.forward(x_test)

    print("\n\n----------------------------------------------------------------------------------")

    print("Let's try our testing data!\n")

    test_accuracy = model.accuracy(predictions, y_test)
    print(f'Accuracy: %{test_accuracy * 100:.3f}')

    print("\n\n----------------------------------------------------------------------------------")



    



