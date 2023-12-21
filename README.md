# Phishing Website Detection

This repository contains code for a machine learning model to detect phishing websites based on various features. The dataset used in this analysis is stored in the file `Phishing.csv`. The model is implemented using the RandomForestClassifier from scikit-learn.

## Dataset Description

The dataset contains information about different websites with 49 features. The features include numerical and binary attributes related to the structure and content of the URLs. The target variable, `CLASS_LABEL`, indicates whether a website is classified as phishing (1) or not (0).

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

### Installation

Clone the repository and install the required libraries:

```bash
git clone https://github.com/your-username/phishing-detection.git
cd phishing-detection
pip install -r requirements.txt
```

## Exploratory Data Analysis

- The dataset is loaded using pandas, and basic information such as the shape, missing values, and descriptive statistics are explored.

## Feature Selection

- Irrelevant feature ('id') is dropped from the dataset.

## Data Visualization

- A heatmap is generated to visualize the correlation between different features in the dataset.

## Model Training and Evaluation

- The RandomForestClassifier is trained on a subset of features selected for the analysis.
- The model is evaluated using accuracy scores on both the training and test sets.
- The trained model is saved as a pickle file (`Phishing.pickle`) for future use.

## Results

The RandomForestClassifier achieves an accuracy of 99.81% on the training set and 97.85% on the test set, indicating good performance in detecting phishing websites.

## Usage

To use the trained model for predictions, load the pickle file and call the `predict` method on new data.

```python
import pickle

# Load the saved model
loaded_model = pickle.load(open('Phishing.pickle', 'rb'))

# Make predictions on new data
new_data = ...  # Prepare your new data
predictions = loaded_model.predict(new_data)
```

