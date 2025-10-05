# Amazon: Multi-Class Product Classification

## Goal
Given a query and a result list of products retrieved for this query, classify each product as being an **Exact**, **Substitute**, **Complement**, or **Irrelevant** match for the query.  

## Members
- Arya Mhaiskar 
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

## Preprocessing
1. Data cleaning
    1. Merged datasets, removed unnecessary columns
    2. Converted remaining columns to lowercase
    3. Removed HTML tags and non-alphanumeric characters
    4. Removed stopwords
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
     1. TD-IDF and Count Vectorier for language-independent tokenization
     2. Trained the model on tokenized training data

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
* Saved preprocessed, merged dataaet as a separate file we could load in our modeling notebook without wasting available RAM
- With a more powerful local environment, or by purchasing more GPU units on Colab, we can further improve our models' performance by training on more data
