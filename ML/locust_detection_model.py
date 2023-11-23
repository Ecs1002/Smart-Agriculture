import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.feature import hog
from skimage import exposure
import joblib

# System libraries
from pathlib import Path
from PIL import Image
import os.path

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
dataset_path = '/kaggle/input/pest-normalized'

image_dir = Path(dataset_path)

# Get filepaths and labels
print("Getting filepaths and labels...")
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpeg')) + list(image_dir.glob(r'**/*.png'))
print("Filepaths and labels retrieved successfully.")

# Check if any files were found
if not filepaths:
    print(f"No image files found in directory {image_dir}")
else:
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

# Convert filepaths and labels to pandas Series
print("Converting filepaths and labels to pandas Series...")
filepaths_series = pd.Series(filepaths, name='Filepath')
labels_series = pd.Series(labels, name='Label')
print("Filepaths and labels converted to pandas Series successfully.")

# Concatenate filepaths and labels
print("Concatenating filepaths and labels...")
image_df = pd.concat([filepaths_series, labels_series], axis=1)
print("Filepaths and labels concatenated successfully.")

# Display 16 pictures of the dataset with their labels
random_index = np.random.randint(0, len(image_df), 16)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    image = Image.open(image_df.Filepath[random_index[i]])
    ax.imshow(image)
    ax.set_title(image_df.Label[random_index[i]])
plt.tight_layout()
plt.show()

train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
test_df['Label'] = label_encoder.transform(test_df['Label'])
print("Labels encoded successfully.")

# Save the LabelEncoder
print("Saving the LabelEncoder...")
joblib.dump(label_encoder, 'label_encoder.pkl')
print("LabelEncoder saved successfully.")

# Function to extract HOG features from an image
def extract_hog_features(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    
    # Calculate HOG features and return flattened array
    features, _ = hog(img_array, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
    return features

# Extract HOG features for training data
print("Extracting HOG features for training data...")
X_train = np.array([extract_hog_features(filepath) for filepath in train_df['Filepath']])
print("HOG features extracted for training data.")

# Extract HOG features for test data
print("Extracting HOG features for test data...")
X_test = np.array([extract_hog_features(filepath) for filepath in test_df['Filepath']])
print("HOG features extracted for test data.")

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Create a Random Forest classifier
print("Creating a Random Forest classifier...")
model = RandomForestClassifier()
print("Random Forest classifier created successfully.")

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

# Fit the GridSearchCV object to the data
print("Fitting GridSearchCV object to the data...")
grid_search.fit(X_train, train_df['Label'])
print("GridSearchCV object fitted successfully.")

# Print the best parameters
print(grid_search.best_params_)

# Evaluate the model
print("Evaluating the model...")
y_pred = grid_search.predict(X_test)

y_test = test_df['Label']  # Define y_test

print(classification_report(y_test, y_pred))
print("Model evaluated successfully.")

# Save the model
print("Saving the model...")
filename = joblib.dump(grid_search, 'model.pkl')
print(f"Model saved as {filename}")
