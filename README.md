# NLP TUTORIAL PART 1


1. [Data Cleaning](#data-cleaning)
2. [Tokenization](#tokenization)
3. [Stopwords Removal](#stopwords-removal)
4. [Stemming and Lemmatization](#stemming-and-lemmatization)
5. [POS Tagging](#pos-tagging)
6. [Named Entity Recognition (NER)](#named-entity-recognition-ner)
7. [WordNet](#wordnet)
8. [Corpus Methods](#corpus-methods)
9. [Frequency Distribution](#frequency-distribution)
10. [Collocations](#collocations)
11. [RegexpParser](#regexpparser)
12. [TextBlob](#textblob)
13. [WordCloud](#wordcloud)


# NLP TUTORIAL PART 2


## 1. [BAG OF WORDS](#bag-of-words)
- 1.1 [Tokenization](#tokenization)
- 1.2 [Vocabulary](#vocabulary)
- 1.3 [Vectorization](#vectorization)
  - 1.3.1 [Code with Examples](#code-with-examples)
- 1.4 [Pros and Cons of Bag of Words (BoW) Model](#pros-and-cons-of-bag-of-words-bow-model)

## 2. [BAG OF WORDS HYPERPARAMETERS](#bag-of-words-hyperparameters)
- 2.1 [N-grams](#n-grams)
  - 2.1.1 [Unigrams](#unigrams)
  - 2.1.2 [Bigrams](#bigrams)
  - 2.1.3 [Trigrams](#trigrams)
- 2.2 [Pros and Cons of N-grams in Natural Language Processing (NLP)](#pros-and-cons-of-n-grams-in-natural-language-processing-nlp)

## 3. [TF-IDF](#tf-idf)
- 3.1 [Code Examples](#code-examples)
- 3.2 [TF-IDF Pros and Cons](#tf-idf-pros-and-cons)

## 4. [Custom Features in Natural Language Processing (NLP)](#custom-features-in-natural-language-processing-nlp)

## 5. [Mini Project IMDB Dataset of 50K Movie Reviews](#mini-project-imdb-dataset-of-50k-movie-reviews)
- 5.1 [Define library](#define-library)
- 5.2 [Dataset Download from Gdrive](#dataset-download-from-gdrive)
- 5.3 [LOAD DATA](#load-data)
- 5.4 [UNDERSTAND DISTRIBUTION OF OUTPUT COLUMN](#understand-distribution-of-output-column)

## 6. [Data Preprocessing](#data-preprocessing)
- 6.1 [DESCRIBE, HEAD, ISNULL, INFO, DUPLICATE](#describe-head-isnull-info-duplicate)
- 6.2 [BEFORE PROCESSING WILL UNDERSTAND THE DATASET BY FOLLOWING QUESTIONS](#before-processing-will-understand-the-dataset-by-following-questions)
  - 6.2.1 [Find out the Corpus total size?](#find-out-the-corpus-total-size)
  - 6.2.2 [Find out the document char size and word size?](#find-out-the-document-char-size-and-word-size)
  - 6.2.3 [Find out the bigrams and trigrams?](#find-out-the-bigrams-and-trigrams)
  - 6.2.4 [Find out the vocabulary?](#find-out-the-vocabulary)

## 7. [NLP Preprocessing Workflow Index](#nlp-preprocessing-workflow-index)
- 7.1 [Inspecting Data](#inspecting-data)
- 7.2 [Cleaning](#cleaning)
  - 7.2.1 [First Will Clean the Data Using Regular Expression](#first-will-clean-the-data-using-regular-expression)
    - 7.2.1.1 [HTML Tag Removal](#html-tag-removal)
    - 7.2.1.2 [Emoji Removal](#emoji-removal)
    - 7.2.1.3 [Lowercasing](#lowercasing)
    - 7.2.1.4 [Text Analyzer](#text-analyzer)
    - 7.2.1.5 [Spelling Correction (PySpellchecker, TextBlob)](#spelling-correction-pyspellchecker-textblob)
- 7.3 [Difference between PySpellChecker and TextBlob](#difference-between-pyspellchecker-and-textblob)
- 7.4 [Basic Preprocessing](#basic-preprocessing)
  - 7.4.1 [Tokenization](#tokenization-1)
    - 7.4.1.1 [Sentence Tokenization](#sentence-tokenization)
    - 7.4.1.2 [Word Tokenization](#word-tokenization)
- 7.5 [Advance Preprocessing](#advance-preprocessing)
  - 7.5.1 [Stopwords Removal](#stopwords-removal)
  - 7.5.2 [Stemming](#stemming)
  - 7.5.3 [Stemming Vs Lemmatization](#stemming-vs-lemmatization)
  - 7.5.4 [Lemmatization](#lemmatization)
  - 7.5.5 [Remove Digits and Punctuation](#remove-digits-and-punctuation)
- 7.6 [Frequency Distribution](#frequency-distribution)
- 7.7 [Language Detection](#language-detection)

## 8. [Model Building](#model-building)
- 8.1 [LABEL ENCODING FOR TARGET](#label-encoding-for-target)
- 8.2 [Multinomial Naive based algorithm](#multinomial-naive-based-algorithm)
- 8.3 [Bag of Words](#bag-of-words-1)
- 8.4 [Train-test split](#train-test-split)
- 8.5 [MODEL FIT](#model-fit)
  - 8.5.1 [Model training](#model-training)
  - 8.5.2 [Model evaluation](#model-evaluation)
    - 8.5.2.1 [Accuracy, Precision, Recall, F1-score](#accuracy-precision-recall-f1-score)
    - 8.5.2.2 [Confusion matrix](#confusion-matrix)


## Tasks

### 1. Data Cleaning <a name="data-cleaning"></a>
- Convert text to lowercase.
- Remove punctuation using the `string` library or regular expressions (`re`).

### 2. Tokenization <a name="tokenization"></a>
- Sentence tokenize using NLTK.
- Word tokenize using NLTK.

### 3. Stopwords Removal <a name="stopwords-removal"></a>
- Use NLTK's stopwords corpus to remove stopwords.

### 4. Stemming and Lemmatization <a name="stemming-and-lemmatization"></a>
- Use NLTK's `PorterStemmer`, `LancasterStemmer`, and `WordNetLemmatizer`.
- Use spaCy's lemmatization.

### 5. POS Tagging <a name="pos-tagging"></a>
- Perform Part of Speech (POS) tagging using NLTK or spaCy.

### 6. Named Entity Recognition (NER) <a name="named-entity-recognition-ner"></a>
- Utilize NER using NLTK's `ne_chunk` function or spaCy.

### 7. WordNet <a name="wordnet"></a>
- Access WordNet's synsets, definitions, and lemma names.

### 8. Corpus Methods <a name="corpus-methods"></a>
- Explore corpus methods available in NLTK.

### 9. Frequency Distribution <a name="frequency-distribution"></a>
- Use NLTK's `FreqDist` function and plot the distribution.

### 10. Collocations <a name="collocations"></a>
- Identify collocations in text using NLTK.

### 11. RegexpParser <a name="regexpparser"></a>
- Utilize NLTK's `RegexpParser` for parsing based on regular expressions.

### 12. TextBlob <a name="textblob"></a>
- Analyze text using TextBlob.

### 13. WordCloud <a name="wordcloud"></a>
- Generate word clouds from text.
