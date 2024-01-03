import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# Set random seed for reproducible randomness
np.random.seed(42)

# Returns 100 evenly spaced numbers between 0 and 1
X_pretransform = np.linspace(0, 1, 100)

# Apply a non-linear transformation to the input features
X = np.sin(1 * np.pi * X_pretransform) + np.sin(5 * np.pi * X_pretransform) + np.sin(10 * np.pi * X_pretransform)

# The output variable is linearly related to the explanatory variable X, with added noise. The np.random.randn() functio returns a value from the standard normal distribution. *X.shape unpacks the tuple and returns the shape of X and it is inputted so that for every value in X there is a random noise, and we multiply this by 0.3
y = 3.5 * X_pretransform + np.random.randn(*X.shape) * 0.05

# Split the data into training and validation sets with 20% of the data used for validation. The random_state parameter is set to 42 so that the results are reproducible.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# reshape(-1, 1) in Python reshapes an array to have one column and as many rows as needed to fit all the data. Think of it like taking a line of people and rearranging them into a single-file vertical line. The -1 tells the computer to figure out how many rows there will be automatically. We reshape `X_train` to a two-dimensional array because machine learning models in libraries like TensorFlow expect input data in a specific format where each example is in its own row and each feature is in its own column, even if there's just one feature. Reshaping ensures the data is structured the way the model requires it to be.

# training dataset
train_features = tf.convert_to_tensor(X_train.reshape(-1, 1), dtype=tf.float32)
#training target set
train_labels = tf.convert_to_tensor(y_train.reshape(-1, 1), dtype=tf.float32)

# validation dataset
val_features = tf.convert_to_tensor(X_val.reshape(-1, 1), dtype=tf.float32)
# validation target set
val_labels = tf.convert_to_tensor(y_val.reshape(-1, 1), dtype=tf.float32)

# Linear regression model with one input neuron, 3 neurons in one fully connected hidden layer, and one output neuron. The input shape is 1 because there is only one feature in the dataset. The activation function for the hidden layer is relu, and the activation function for the output layer is linear.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=3, input_shape=[1]),
    tf.keras.layers.Dense(units=1)
])

# The optimizer is stochastic gradient descent (sgd) and the loss function is mean squared error (mse).
model.compile(optimizer='sgd', loss='mean_squared_error')


history = model.fit(train_features, train_labels, epochs=100, validation_data=(val_features, val_labels))

# Select 5 random indices from the validation set
random_indices = np.random.choice(len(X_val), 5, replace=False)
sample_val_features = tf.convert_to_tensor(X_val[random_indices].reshape(-1, 1), dtype=tf.float32)
sample_val_labels = y_val[random_indices]

# Make predictions on the selected samples
predictions = model.predict(sample_val_features)

# Print the actual vs predicted values
for i, index in enumerate(random_indices):
    print(f"Input: {sample_val_features[i][0].numpy()}")
    print(f"Actual Output: {sample_val_labels[i]}")
    print(f"Predicted Output: {predictions[i][0]}")
    print("-------------")