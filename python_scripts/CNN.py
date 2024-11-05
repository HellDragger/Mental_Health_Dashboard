import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Paths for train and test data
train_dir = '../data/image_training/train/'
test_dir = '../data/image_training/test/'

# Model save path
model_save_path = '../image_model.h5'

# Define image parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 10

# Emotion categories
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad']

# Depression, Anxiety, Anger, and Normal combination logic
def get_emotion_scores(predictions):
    depression_labels = ['sad', 'fearful', 'disgusted']
    anxiety_labels = ['sad', 'fearful']
    anger_label = 'angry'
    normal_labels = ['happy', 'neutral']

    depression_score = sum([predictions[label] for label in depression_labels])
    anxiety_score = sum([predictions[label] for label in anxiety_labels])
    anger_score = predictions[anger_label]
    normal_score = sum([predictions[label] for label in normal_labels])

    # Normalize the scores to sum up to 100
    total = depression_score + anxiety_score + anger_score + normal_score
    if total > 0:
        depression_score = (depression_score / total) * 100
        anxiety_score = (anxiety_score / total) * 100
        anger_score = (anger_score / total) * 100
        normal_score = (normal_score / total) * 100

    return {
        'depression': depression_score,
        'anxiety': anxiety_score,
        'anger': anger_score,
        'normal': normal_score
    }

# Data augmentation and data loading
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10, zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# CNN Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(EMOTIONS), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)

# Save the trained model to the root directory
model.save(model_save_path)
print(f'Model saved at {model_save_path}')

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# Get predictions
test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification Report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Test emotion score calculation
def calculate_emotion_scores(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    emotion_probs = {emotion: prediction[i] for i, emotion in enumerate(class_labels)}

    scores = get_emotion_scores(emotion_probs)
    return scores

# Example usage of emotion score calculation for a test image
example_image_path = test_dir + 'angry/im0.png'  # Replace with an actual test image path
scores = calculate_emotion_scores(example_image_path)
print(f"Emotion scores for the test image: {scores}")

# Plotting training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
