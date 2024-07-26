import tensorflow as tf 
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense, LSTM  
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
import numpy as np

from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# Load the data from the pickle file
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Get the data and labels
data = np.asarray(data_dict['data'], dtype=np.float32)
labels = np.asarray(data_dict['labels'], dtype=np.int32)

# Reshape data to 2D array (samples, height, width, channels)
data = data.reshape((data.shape[0], 1, data.shape[1]))  # Treating the data as a "21x2" image with 1 channel

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Build the CNN model
model = Sequential()
model.add(LSTM(64, input_shape=(1, 42), activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# predict the labels for the test set
y_predict = model.predict(X_test)

# Convert probabilities to class labels
predicted_labels = np.argmax(y_predict, axis=1)

# Print the predicted labels
print(predicted_labels)

# save the model in a pickle file 
f = open('neuralmodel.pickle', 'wb')

# dump the model as a key value pair in the pickle file 
pickle.dump({'model': model}, f)
f.close()
