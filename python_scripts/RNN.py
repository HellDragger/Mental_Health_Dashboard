import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import nltk
import os

nltk.download('punkt')

# Load and concatenate three datasets
df1 = pd.read_csv('../data/training data/goemotions_1.csv')
df2 = pd.read_csv('../data/training data/goemotions_2.csv')
df3 = pd.read_csv('../data/training data/goemotions_3.csv')

# Concatenate the datasets
df = pd.concat([df1, df2, df3], ignore_index=True)

# Function to map emotions to Depression, Anxiety, Anger, and Normal/Neutral
def map_emotions(row):
    depression_emotions = ['sadness', 'disappointment', 'grief', 'remorse', 'disgust', 'embarrassment', 'fear', 'neutral']
    anxiety_emotions = ['fear', 'nervousness', 'confusion', 'annoyance', 'disgust', 'embarrassment', 'curiosity', 'surprise']
    anger_emotions = ['anger']
    positive_emotions = ['admiration', 'amusement', 'approval', 'caring', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief']
    
    # Depression Score
    depression_score = row[depression_emotions].sum()
    
    # Anxiety Score
    anxiety_score = row[anxiety_emotions].sum()
    
    # Anger Score
    anger_score = row['anger']  # Already directly available
    
    # Normal/Neutral Score (collect all positive emotions)
    normal_score = row[positive_emotions].sum()
    
    return pd.Series([depression_score, anxiety_score, anger_score, normal_score])

# Filter only the relevant columns (assuming the GoEmotions dataset has these emotion columns)
emotions_df = df[['text', 'sadness', 'disappointment', 'grief', 'remorse', 'disgust', 'embarrassment', 'fear', 'neutral',
                  'nervousness', 'confusion', 'annoyance', 'curiosity', 'surprise', 'anger', 'admiration', 'amusement',
                  'approval', 'caring', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief']]

# Apply the mapping to the dataset to create new columns for Depression, Anxiety, Anger, and Normal
emotions_df[['depression', 'anxiety', 'anger', 'normal']] = emotions_df.apply(map_emotions, axis=1)

# Preprocessing text data
tokenizer = Tokenizer(num_words=10000)  # Set max number of words
tokenizer.fit_on_texts(emotions_df['text'])
sequences = tokenizer.texts_to_sequences(emotions_df['text'])

# Padding sequences to have the same length
max_len = 100
X = pad_sequences(sequences, maxlen=max_len)

# Labels (convert to a scale of 0-100)
y = emotions_df[['depression', 'anxiety', 'anger', 'normal']] * 100

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Definition
model = Sequential()

# Embedding layer
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_len))

# LSTM layer
model.add(LSTM(units=128, return_sequences=False))

# Fully connected (dense) layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output layer - 4 units for Depression, Anxiety, Anger, and Normal scores (0-100)
model.add(Dense(4, activation='linear'))  # 'linear' for regression (0-100 scores)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Save the model for future use
model.save('../text_model.h5')

# Evaluate the model on the test data
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Function to load text files from multiple folders
def load_text_files_from_folders(parent_directory):
    texts = []
    # Walk through each folder in the parent directory
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    return texts

# Function to predict emotions from a list of texts
def predict_emotions(texts):
    # Preprocess the texts
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    # Predict using the trained model
    raw_predictions = model.predict(padded_sequences)
    
    # Normalize the predictions so that the sum of all emotion scores equals 100
    normalized_predictions = []
    for prediction in raw_predictions:
        total = np.sum(prediction)  # Sum of all predicted scores
        normalized_prediction = (prediction / total) * 100  # Scale each score to make the sum 100
        normalized_predictions.append(normalized_prediction)
    
    return np.array(normalized_predictions)

# Path to the parent directory where folders of text files are stored
parent_directory_path = '../data/instagram_downloads'  # Replace with the actual parent directory path

# Load text files from all folders inside the parent directory
texts = load_text_files_from_folders(parent_directory_path)

# Predict emotions on the text files
emotion_scores = predict_emotions(texts)

# Display the results
for i, text in enumerate(texts):
    print(f"Text {i+1}: {text[:100]}...")
    print(f"Depression: {emotion_scores[i][0]:.2f}, Anxiety: {emotion_scores[i][1]:.2f}, Anger: {emotion_scores[i][2]:.2f}, Normal: {emotion_scores[i][3]:.2f}\n")
