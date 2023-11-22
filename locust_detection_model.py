import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import joblib

# System libraries
from pathlib import Path
from PIL import Image
import os.path

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
dataset_path = r'C:\Users\deepa\Downloads\locust_pics'

image_dir = Path(dataset_path)

# Get filepaths and labels
print("Getting filepaths and labels...")
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpeg')) + list(image_dir.glob(r'**/*.png'))

# Check if any files were found
if not filepaths:
    print(f"No image files found in directory {image_dir}")
else:
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

# Convert filepaths and labels to pandas Series
filepaths_series = pd.Series(filepaths, name='Filepath')
labels_series = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
print("Concatenating filepaths and labels...")
image_df = pd.concat([filepaths_series, labels_series], axis=1)

print(len(list(image_dir.glob(r'**/*.jpeg'))))

image_df

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

# Prepare features and labels
print("Preparing features and labels...")
X_train = np.array([np.array(Image.open(filepath).convert('RGB').resize(IMAGE_SIZE)) for filepath in train_df['Filepath']])
y_train = train_df['Label']

X_test = np.array([np.array(Image.open(filepath).resize(IMAGE_SIZE)) for filepath in test_df['Filepath']])

# Create a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)

# Train the model
print("Training the model...")
model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
print("Model trained successfully.")

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

y_test = test_df['Label']  # Define y_test

print(classification_report(y_test, y_pred))
print("Model evaluated successfully.")

if os.access(os.getcwd(), os.W_OK):
    print("You have write permission for this directory.")
else:
    print("You do not have write permission for this directory.")

# Display results
print("Displaying results...")
random_index = np.random.randint(0, len(test_df) - 1, 25)
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 20),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    image = Image.open(test_df.Filepath.iloc[random_index[i]])
    ax.imshow(image)
    if test_df.Label.iloc[random_index[i]] == y_pred[random_index[i]]:
        color = "green"
    else:
        color = "red"
    ax.set_title(f"True: {test_df.Label.iloc[random_index[i]]}\nPredicted: {y_pred[random_index[i]]}", color=color)
plt.show()
plt.tight_layout()
print("Results displayed successfully.")

def predict_insect(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB').resize((224, 224))  # resize to (224, 224)
    img_array = np.array(img).reshape(1, -1)

    # Use the model to predict the class of the image
    print("Predicting the class of the image...")
    pred = model.predict(img_array)

    # Convert the predicted class back to the original label
    pred_label = label_encoder.inverse_transform(pred)

    return pred_label[0]

# Save the model
print("Saving the model...")
filename = joblib.dump(model, 'model.pkl')
print(f"Model saved as {filename}")
