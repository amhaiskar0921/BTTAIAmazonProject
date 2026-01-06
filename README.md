# Amazon: Multi-Class Product Classification

## Goal
Given a query and a result list of products retrieved for this query, classify each product as being an **Exact**, **Substitute**, **Complement**, or **Irrelevant** match for the query.  

## Members
- Arya Mhaiskar (Lead)
- Rayyan Ashraf
- Lucy Yin
- Isabella Qian
- Lily Zhou

**Challenge Advisor**: Chen Luo, Sr. Applied Scientist at Amazon Search  
**Teaching Assistant:** Vaibhav Tiwari

## Datasets (Large-scale)
**Dataset 1:** Labeled dataset with user search query to product ID mappings  
**Dataset 2:** Product metadata including the product ID, title, description, brand, and color
**Languages:** User search queries and product metadata in *English*, *Spanish*, and *Japanese*  
**Source:** [Task 2 from Amazon KDD Cup'22 Challenge](https://github.com/amazon-science/esci-data/tree/main)  
**Class frequencies:**
- 65.17% Exacts, 21.91% Substitutes, 2.89% Complements and 10.04% Irrelevants
- 13.6% queries and products in Spanish, 17.0% in Japanese, 69.4% in English

## Preprocessing
1. Data cleaning
    1. Merged datasets, removed unnecessary columns like the query ID
    3. Converted remaining columns to lowercase
    4. Removed HTML tags and non-alphanumeric characters like emojis across all 3 languages
    6. Removed stopwords
2. Data preprocessing
    1. Stemming
         1. nltk PorterStemmer for English, SnowballStemmer for Spanish
         2. MeCab for Japanese
    2. Lemmatizing
         1. nltk WordNetLemmatizer for English, spaCy for Spanish

## Modeling
1. BERT Base Multilingual Cased
     1. Same model used for language-independent tokenization
     2. Created TensorFlow Dataset objects for training and testing
     3. Fine-tuned the model with training data
2. Logistic Regression
     1. TF-IDF and Count Vectorizer for language-independent tokenization
     2. Trained the model on tokenized training data
 3. Random Forest (Added later)
     1. TF-IDF for language-independent tokenization
     2. Trained the model on tokenized training data
     3. The model had decent accuracy (64%), but low F1-score since it only guessed the majority class correctly
 3. Complement Naive Bayes (Added later)
     1. TF-IDF for language-independent tokenization
     2. The Complement Naive Bayes (CNB) variant of the Naive Bayes algorithm is good for imbalanced class datasets and text classification
     3. Trained the model on tokenized training data
     4. The model had an F1-score of 0.40
        
## Evaluation
**Used micro-averaging F1-score to account for class imbalance in the label classes**  
F1-scores:  
<img width="1000" alt="image" src="https://github.com/amhaiskar0921/BTTAIAmazonProject/assets/43621944/b133af27-cef3-4861-81aa-2e9efbb543c7"> Baseline BERT scores are from the bert-base-multiligual-cased prior to fine-tuning

### Interpretation
* Logistic Regression (pure ML approach) and fine-tuned BERT perform equally well for this task (**this was a novel, unexpected finding**)
* Logistic Regression takes considerably less training time and memory than BERT
<img width="1000" alt="image" src="https://github.com/amhaiskar0921/BTTAIAmazonProject/assets/43621944/32e7f246-555e-4b71-b006-2b20f4c9684f">

## Limitations of using Google Colab
- **GPU unit limits in the free tier**  
* We had several RAM overflow issues in the training and preprocessing phases. We mitigated these through random sampling from our large-scale datasets and separating preprocessing and modeling in separate notebooks
* Saved the preprocessed, merged dataset as a separate file we could load in our modeling notebook without wasting available RAM
- With a more powerful local environment, or by purchasing more GPU units on Colab, we can further improve our models' performance by training on more data
## Next Steps
1. Experiment with more models (added Random Forest and Complement Naive Bayes above)
2. Create a more balanced dataset to train the models
