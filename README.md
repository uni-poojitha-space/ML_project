**News Article Category Classification Project**

This project focuses on building a machine learning model to classify news articles into predefined categories based on their titles and content. The dataset contains 6,877 unique values across three columns: category, title, and body of news articles. This dataset is crucial for understanding trends in news reporting, identifying popular topics, and analyzing the sentiment and tone of articles across different categories. By examining the content and categorization of news articles, researchers and businesses can gain insights into public interests, media biases, and the evolution of news narratives over time. This information can be leveraged for various applications, including media analysis, content recommendation systems, and sentiment analysis.

**Project Overview**

The goal of this project is to classify news articles into predefined categories using machine learning techniques. Classifying news articles is a crucial task in content management systems, enabling automated categorization for improved user experience and content delivery. By accurately categorizing articles, users can easily access relevant content, enhancing their overall engagement and satisfaction. 

**ML Models Used**
Logistic Regression
Support Vector Machine (SVM)
Random Forest
Naive Bayes
Gradient Boosting
Stacking Models

**Preprocessing**

Various preprocessing techniques like text normalization, tokenization, and Term Frequency-Inverse Document Frequency (TF-IDF) vectorization were used to convert the raw text data into structured format suitable for modeling. Hyperparameter tuning was performed on each model to optimize performance.
In the preprocessing steps for the dataset, I performed the following tasks:
Text Cleaning: Removing unnecessary characters, punctuation, and numbers from the headlines and descriptions.
Tokenization: Splitting the text into individual words or tokens.
Lowercasing: Converting all text to lowercase to ensure uniformity.
Stopword Removal: Eliminating common words that do not contribute to the classification task (e.g., "and," "the").
Stemming/Lemmatization: Reducing words to their base form to group similar words together (e.g., "running" to "run").

**Tools and Libraries Used**
Python
Scikit-learn
Pandas
Numpy
Matplotlib
Seaborn
NLTK for text processing
Google Colab for execution

**
Getting Started
Prerequisites**

Python 3.x
Install the required libraries using pip install -r requirements.txt

**Running the Project**

Clone the repository.
git clone https://github.com/yourusername/news-classification.git

**Run the Jupyter notebook for training and evaluation:**

jupyter notebook MLfinal_Project_5.ipynb

**Results**
The models were trained and metrics like accuracy, precision, and F1 score were used to assess their performance. No single model was significantly superior across all categories, and a stacked approach was used to balance the strengths of different models.

**Conclusion**
This project demonstrates how various machine learning techniques can be applied to classify news articles accurately, showcasing the importance of model selection, tuning, and ensemble methods in tackling real-world text classification tasks.

Links

Colab Notebook: [Link to Colab](https://colab.research.google.com/drive/1kGsJ6GCdtX7ry8Efy5s9Y0j9A4mt_qDd?usp=sharing)
Dataset Source: [Link to Dataset](https://www.kaggle.com/code/whdhdyt/news-article-category-classification/input)
