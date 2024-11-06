
# Social Media Emotional Dashboard (SMED)

This project, **SMED**, is a tool designed to analyze the emotional content of Instagram posts. It uses machine learning models to classify emotions and displays the results on a user-friendly dashboard.

## Project Overview
The **Social Media Emotional Dashboard** leverages the **GoEmotions dataset** and two Kaggle datasets to train models that analyze emotions in Instagram posts. The dashboard allows users to input an Instagram URL and the number of posts they wish to analyze, displaying emotion-based insights after processing the data.

## Project Structure

```plaintext
SMED/
│
├── datasets/
│   ├── train/     # Training dataset
│   ├── test/      # Testing dataset
│
├── python_scripts/
│   ├── Instagram.py      # Instagram data extraction script
│   ├── text_extraction.py # Text processing and extraction
│   ├── RNN.py            # RNN model script for emotion classification
│   ├── CNN.py            # CNN model script for emotion classification
│
├── app.py                # Main app file for launching the dashboard
├── requirements.txt      # List of required Python libraries
├── README.md             # Project readme
└── ...

Installation

1. Download the Datasets

GoEmotions Dataset:

To download the GoEmotions dataset, run the following command in the command prompt:

git clone https://github.com/google-research/google-research/tree/master/goemotions

Kaggle Datasets:

Download the following Kaggle datasets and place them in the appropriate train/ and test/ directories:

	•	Kaggle Dataset 1
	•	Kaggle Dataset 2

Ensure the datasets follow the existing folder structure in datasets/train/ and datasets/test/.

2. Create a Virtual Environment

Before installing the required libraries, create a virtual environment to isolate the project dependencies. Run the following command:

python -m venv env

Activate the virtual environment:

	•	For Windows:

.\env\Scripts\activate


	•	For macOS/Linux:

source env/bin/activate



3. Install Required Libraries

With the virtual environment activated, install the necessary libraries by running:

pip install -r requirements.txt

This will install all the dependencies specified in requirements.txt.

4. Run the Python Scripts

The project contains several Python scripts that should be run sequentially to extract Instagram data, process text, and classify emotions. Run the scripts in the following order:

	1.	Instagram Data Extraction:

python python_scripts/Instagram.py


	2.	Text Extraction:

python python_scripts/text_extraction.py


	3.	RNN Model Training:

python python_scripts/RNN.py


	4.	CNN Model Training:

python python_scripts/CNN.py



5. Run the Dashboard

After running the scripts, launch the dashboard by running:

python app.py

This will start the application and open the dashboard interface in your browser.

Using the Dashboard

	1.	Enter an Instagram URL in the provided input field.
	2.	Specify the number of posts you want to analyze.
	3.	Click the Analyze button.

Once the analysis is complete, the results will be displayed on the dashboard, showing the emotional classification of the posts.