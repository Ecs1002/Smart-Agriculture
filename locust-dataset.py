import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load your dataset
csv_file_path = r'C:\Users\deepa\Downloads\locust.csv'
df = pd.read_csv(csv_file_path)

# Extract pixel values and labels
X = df.iloc[:, :-1].values  # Features (pixel values)
y = df.iloc[:, -1].values    # Labels ('Y/N')

# Data Normalization/Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Assuming 'Y' represents the positive class and 'N' represents the negative class
# Convert labels to numerical format (0 and 1)
y_binary = (y == 'Y').astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,  # Increase the number of epochs if needed
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Predictions on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred_binary)
print("Classification Report:")
print(class_report)

# Save the trained model for future use (e.g., deployment)
model.save('locust_detection_model.h5')

# Additional Steps:
# 2. Implement model deployment for making predictions on new data
#    (this involves loading the saved model and providing new input data)

# 3. Monitor the model's performance over time and retrain as needed
#    (especially if the model is deployed in a real-world scenario)

# 4. Consider techniques for interpreting or explaining model predictions
#    (e.g., SHAP, LIME) for better understanding and transparency.
