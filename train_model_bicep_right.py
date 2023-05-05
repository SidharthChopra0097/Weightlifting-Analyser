from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.regularizers import l2

# Define the neural network architecture
model = Sequential()
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01), input_dim=2))
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
history =model.fit(train_angles, train_labels, validation_data=(val_angles, val_labels), epochs=20, batch_size=10)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_angles, test_labels)
print('Test accuracy:', accuracy)