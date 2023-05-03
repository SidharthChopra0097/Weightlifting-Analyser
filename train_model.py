from keras.models import Sequential
from keras.layers import Dense,Dropout

# Define the neural network architecture
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=2))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
history =model.fit(train_angles, train_labels, validation_data=(val_angles, val_labels), epochs=20, batch_size=5)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_angles, test_labels)
print('Test accuracy:', accuracy)