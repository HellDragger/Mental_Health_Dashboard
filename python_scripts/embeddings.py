import os
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

def load_cleaned_text(file_path):
    """
    Load the cleaned text from the provided file path.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def generate_embeddings(text, model):
    """
    Generate BERT embeddings for the given text using the provided model.
    """
    # Use the model to encode the text and generate embeddings
    embeddings = model.encode(text, convert_to_tensor=True)
    return embeddings

def save_embeddings(embeddings, output_file_path):
    """
    Save the embeddings as a .npy (NumPy) file.
    """
    np.save(output_file_path, embeddings.cpu().numpy())

def generate_text_embeddings(source_folder, output_folder, model):
    """
    Load cleaned text, generate BERT embeddings, and save them to the output folder.
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
                    # Load the cleaned text
                    text = load_cleaned_text(file_path)

                    # Generate embeddings using the BERT-based model
                    embeddings = generate_embeddings(text, model)

                    # Determine the output file path (change file extension to .npy)
                    relative_path = os.path.relpath(root, source_folder)
                    output_file_dir = os.path.join(output_folder, relative_path)

                    # Create directories in the output folder if they don't exist
                    if not os.path.exists(output_file_dir):
                        os.makedirs(output_file_dir)

                    # Save the embeddings
                    output_file_path = os.path.join(output_file_dir, f"{os.path.splitext(file)[0]}_embeddings.npy")
                    save_embeddings(embeddings, output_file_path)

                    print(f"Generated embeddings for {file_path} -> {output_file_path}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

def main():
    # Set your source and output folders
    source_folder = '../data/cleaned_text'
    output_folder = '../data/text_embeddings'

    # Load a pre-trained BERT-based model (SentenceTransformers for convenience)
    model = SentenceTransformer('all-MiniLM-L6-v2')  # This is a small, efficient BERT-based model

    # Generate embeddings for the cleaned text files
    generate_text_embeddings(source_folder, output_folder, model)

if __name__ == "__main__":
    main()
