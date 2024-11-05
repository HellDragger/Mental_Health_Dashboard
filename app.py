from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from models.text_model import analyze_text
from models.image_model import analyze_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    profile_url = request.form.get('profile_url')
    post_count = int(request.form.get('post_count'))

    # Fetch posts (this is a placeholder, replace with actual fetching logic)
    posts = fetch_instagram_posts(profile_url, post_count)

    scores = []
    for post in posts:
        caption = post['caption']
        image_url = post['image_url']

        # Analyze text and image
        text_scores = analyze_text(caption)
        image_scores = analyze_image(image_url)

        # Combine scores
        combined_scores = {
            'depression': (text_scores['depression'] + image_scores['depression']) / 2,
            'anxiety': (text_scores['anxiety'] + image_scores['anxiety']) / 2,
            'anger': (text_scores['anger'] + image_scores['anger']) / 2,
        }
        scores.append(combined_scores)

    # Save scores to CSV
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv('data/emotion_scores.csv', index=False)

    return redirect(url_for('results'))

@app.route('/results')
def results():
    scores_df = pd.read_csv('data/emotion_scores.csv')
    averages = scores_df.mean().to_dict()

    return render_template('results.html', averages=averages)

def fetch_instagram_posts(profile_url, count):
    # Placeholder function: Implement Instagram API fetching logic
    return [{'caption': 'Sample caption', 'image_url': 'http://example.com/image.jpg'}]

if __name__ == '__main__':
    app.run(debug=True)
