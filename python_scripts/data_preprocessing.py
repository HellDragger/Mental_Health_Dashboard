import os
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stop words (if you haven't already)
nltk.download('stopwords')

# Set of English stop words
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Preprocess and clean the text by:
    - Lowercasing
    - Removing special characters
    - Removing stopwords
    - Removing numbers
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters and numbers using regex
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    words = text.split()
    cleaned_text = ' '.join([word for word in words if word not in stop_words])
    
    return cleaned_text

def preprocess_text_files(source_folder, output_folder):
    """
    Load text files, clean them, and save preprocessed text to a new folder.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse through the source folder and process .txt files
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)

                try:
                    # Read the content of the file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()

                    # Clean the text
                    cleaned_text = clean_text(text)

                    # Determine the output file path
                    relative_path = os.path.relpath(root, source_folder)
                    output_file_dir = os.path.join(output_folder, relative_path)

                    # Create directories in the output folder if they don't exist
                    if not os.path.exists(output_file_dir):
                        os.makedirs(output_file_dir)

                    # Save the cleaned text to a new file
                    output_file_path = os.path.join(output_file_dir, file)
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)

                    print(f"Processed and cleaned {file_path} -> {output_file_path}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

def main():
    # Set your source and output folders
    source_folder = '../data/extracted_text'
    output_folder = '../data/cleaned_text'

    # Preprocess and clean the text files
    preprocess_text_files(source_folder, output_folder)

if __name__ == "__main__":
    main()
