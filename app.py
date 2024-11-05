from flask import Flask, render_template, request
import instaloader
import numpy as np
import easyocr
from keras.models import load_model
from PIL import Image
import requests
from io import BytesIO
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load pre-trained models
text_model = load_model('text_model.h5')  # Path to your text model
image_model = load_model('image_model.h5')  # Path to your image model

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify language(s)

# Initialize tokenizer with a max number of words
tokenizer = Tokenizer(num_words=10000)

# Function to preprocess the input text for the text model
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text.lower())  # Simple cleaning
    # Tokenize and pad the text
    input_sequences = tokenizer.texts_to_sequences([text])
    input_data = pad_sequences(input_sequences, maxlen=100)
    return input_data

# Function to process text and predict emotion scores using text model
def process_text(caption):
    input_data = preprocess_text(caption)
    predictions = text_model.predict(input_data)[0]

    return {
        'depression': predictions[0],
        'anxiety': predictions[1],
        'anger': predictions[2],
        'normal': predictions[3]
    }

# Function to extract text from images using EasyOCR
def extract_text_from_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img_array = np.array(img)

    # Use EasyOCR to extract text
    extracted_text = reader.readtext(img_array, detail=0)  # Extract only the text without details
    return ' '.join(extracted_text)  # Join the extracted text list into a single string

# Function to process images and predict emotion scores using image model
def process_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((150, 150))  # Modify this based on your model's expected input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = image_model.predict(img_array)[0]

    return {
        'depression': predictions[0],
        'anxiety': predictions[1],
        'anger': predictions[2],
        'normal': predictions[3]
    }

# Fetch posts from Instagram using instaloader
def fetch_instagram_posts(profile_url, post_count):
    loader = instaloader.Instaloader()
    profile_name = profile_url.split('/')[-2]  # Extract profile name from URL
    profile = instaloader.Profile.from_username(loader.context, profile_name)

    posts = []
    for post in profile.get_posts():
        if len(posts) >= post_count:
            break
        posts.append({
            'caption': post.caption,
            'image_url': post.url
        })
    return posts

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get Instagram URL and post count from form
        insta_url = request.form['insta_url']
        post_count = int(request.form['post_count'])

        # Fetch posts from Instagram
        posts = fetch_instagram_posts(insta_url, post_count)

        # Initialize combined emotion scores
        combined_scores = {'depression': 0, 'anxiety': 0, 'anger': 0, 'normal': 0}

        for post in posts:
            # Analyze caption
            caption_scores = process_text(post['caption'] if post['caption'] else "")

            # Analyze image text (OCR) and image
            extracted_text = extract_text_from_image(post['image_url'])
            ocr_scores = process_text(extracted_text)
            image_scores = process_image(post['image_url'])

            # Average the scores for the post
            post_scores = {
                'depression': (caption_scores['depression'] + ocr_scores['depression'] + image_scores['depression']) / 3,
                'anxiety': (caption_scores['anxiety'] + ocr_scores['anxiety'] + image_scores['anxiety']) / 3,
                'anger': (caption_scores['anger'] + ocr_scores['anger'] + image_scores['anger']) / 3,
                'normal': (caption_scores['normal'] + ocr_scores['normal'] + image_scores['normal']) / 3
            }

            # Update combined scores
            combined_scores['depression'] += post_scores['depression']
            combined_scores['anxiety'] += post_scores['anxiety']
            combined_scores['anger'] += post_scores['anger']
            combined_scores['normal'] += post_scores['normal']

        # Compute average scores across all posts
        combined_scores = {k: v / post_count for k, v in combined_scores.items()}

        return render_template('results.html', scores=combined_scores)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
