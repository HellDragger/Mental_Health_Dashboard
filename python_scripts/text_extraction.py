import os
import easyocr
from PIL import Image

def extract_text_from_images_easyocr(source_folder, output_folder):
    # Initialize the EasyOCR Reader
    reader = easyocr.Reader(['en'], gpu=False)  # You can add more languages if needed

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Walk through all folders and files in the source_folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if the file is an image (you can add more extensions as needed)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)

                try:
                    # Extract text using EasyOCR
                    extracted_text = reader.readtext(image_path, detail=0)  # detail=0 returns only the text
                    
                    # Join the list of text into a single string
                    extracted_text = "\n".join(extracted_text)
                    
                    # Determine output file path based on the image file name
                    relative_path = os.path.relpath(root, source_folder)
                    output_file_dir = os.path.join(output_folder, relative_path)
                    
                    # Create directories in the output folder if they don't exist
                    if not os.path.exists(output_file_dir):
                        os.makedirs(output_file_dir)
                    
                    # Save the extracted text to a .txt file
                    output_file_path = os.path.join(output_file_dir, f"{os.path.splitext(file)[0]}_text.txt")
                    with open(output_file_path, 'w', encoding='utf-8') as text_file:
                        text_file.write(extracted_text)
                    
                    print(f"Extracted text from {image_path} and saved to {output_file_path}")
                
                except Exception as e:
                    print(f"Could not process {image_path}: {e}")

def main():
    # Set your source and output folders
    source_folder = './instagram_downloads'
    output_folder = './extracted_text'

    # Extract text from images using EasyOCR
    extract_text_from_images_easyocr(source_folder, output_folder)

if __name__ == "__main__":
    main()