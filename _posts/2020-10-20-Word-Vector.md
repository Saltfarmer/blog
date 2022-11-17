---
title: "Word Vector"
header :
  teaser: /assets/images/sklearn-head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Python
  - Preprocessing
  - Sklearn

---

Machine learning can't process non-numeric value. Then how to process image or text data ? Before you train your image or text data, you need to transform the data into numeric value first. This is called feature extraction or feature encoding. In this post, i will focus it on text data first. One way to digitize data is what most machine learning enthusiast called **Bag of words**. The bag-of-words model is a way of representing text data when modeling text with machine learning algorithms. The bag-of-words model is simple to understand and implement and has seen great success in problems such as language modeling and document classification.

## Bag of Words

The approach is very simple and flexible, and can be used in a simple of ways for extracting features from documents. A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:

1. A vocabulary of known words.
2. A measure of the presence of known words.

### Example using bag of words model

1. Collect the data

>'This is the first document.'
>
>'This document is the second document.'
>
>'And this is the third one.'
>
>'Is this the first document?'
>
>'This is a document'
>
>'Are those documents is correct?'

2. Design the vocabulary

> 'and', 'are', 'correct', 'document', 'documents', 'first', 'is', 'one', 'second', 'the', 'third', 'this', 'those'

3. Create document vector

The objective is to turn each document of free text into a vector that we can use as input or output for a machine learning model. The simplest scoring method is to mark the presence of words as a boolean value, 0 for absent, 1 for present. The first sentence's vector ('This is the first document.') would looks like this

> 'and' = 0
>
> 'are' = 0
>
> 'correct' = 0
>
> 'document' = 1
>
> 'documents' = 0
>
> 'first' = 1
>
> 'is' = 1
>
> 'one' = 0
>
> 'second' = 0
>
> 'the' = 1
>
> 'third' = 0 
>
> 'this' = 1
>
> 'those' = 0

## Word Counts with CountVectorizer

The `CountVectorizer()` provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary. The vectors returned from a call to `transform()` will be sparse vectors, and you can transform them back to numpy arrays to look and better understand what is going on by calling the `toarray()` function. Below is an example of using the `CountVectorizer()` to tokenize, build a vocabulary, and then encode a document.

```python
import pandas as pd
df = pd.DataFrame(corpus)
df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>This is the first document.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>This document is the second document.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>And this is the third one.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Is this the first document?</td>
    </tr>
    <tr>
      <th>4</th>
      <td>This is a document</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Are those documents is correct?</td>
    </tr>
  </tbody>
</table>

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x = pd.DataFrame(data=cv.fit_transform(df[0]).toarray(), columns=cv.get_feature_names())
x
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>and</th>
      <th>are</th>
      <th>correct</th>
      <th>document</th>
      <th>documents</th>
      <th>first</th>
      <th>is</th>
      <th>one</th>
      <th>second</th>
      <th>the</th>
      <th>third</th>
      <th>this</th>
      <th>those</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

`toarray()` is to transform the vectorize value into array-like shape. This will make vectorized features is understandable and can be transformed into table in pandas dataframe. The reason why using `df[0]` because simply i just want to vectorize that one single column.

## N-Grams

A more sophisticated approach is to create a vocabulary of grouped words. This both changes the scope of the vocabulary and allows the bag-of-words to capture a little bit more meaning from the document. In this approach, each word or token is called a “gram”. Creating a vocabulary of two-word pairs is, in turn, called a bigram model. Again, only the bigrams that appear in the corpus are modeled, not all possible bigrams. For example of using n-grams (especially bigrams) in `CountVectorizer()`

```python
cv_2 = CountVectorizer(ngram_range=(2, 2))
x = pd.DataFrame(data=cv_2.fit_transform(df[0]).toarray(), columns=cv_2.get_feature_names())
x
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>and this</th>
      <th>are those</th>
      <th>document is</th>
      <th>documents is</th>
      <th>first document</th>
      <th>is correct</th>
      <th>is document</th>
      <th>is the</th>
      <th>is this</th>
      <th>second document</th>
      <th>the first</th>
      <th>the second</th>
      <th>the third</th>
      <th>third one</th>
      <th>this document</th>
      <th>this is</th>
      <th>this the</th>
      <th>those documents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

## Word Frequencies with TFIDF

One issue with simple counts is that some words like “*the*” will appear many times and their large counts will not be very meaningful in the encoded vectors. An alternative is to calculate word frequencies, and by far the most popular method is called TF-IDF. This is an acronym than stands for “*Term Frequency – Inverse Document*” Frequency which are the components of the resulting scores assigned to each word.

- **Term Frequency**: This summarizes how often a given word appears within a document.
- **Inverse Document Frequency**: This downscales words that appear a lot across documents.

Without going into the math, TF-IDF are word frequency scores that try to highlight words that are more interesting, e.g. frequent in a document but not across documents. For example

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
x = pd.DataFrame(data=tfidf.fit_transform(df[0]).toarray(), columns=tfidf.get_feature_names())
x
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>and</th>
      <th>are</th>
      <th>correct</th>
      <th>document</th>
      <th>documents</th>
      <th>first</th>
      <th>is</th>
      <th>one</th>
      <th>second</th>
      <th>the</th>
      <th>third</th>
      <th>this</th>
      <th>those</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.437849</td>
      <td>0.000000</td>
      <td>0.605204</td>
      <td>0.327616</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.437849</td>
      <td>0.000000</td>
      <td>0.378118</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.661292</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.247402</td>
      <td>0.000000</td>
      <td>0.557338</td>
      <td>0.330646</td>
      <td>0.000000</td>
      <td>0.285539</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.512216</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.227372</td>
      <td>0.512216</td>
      <td>0.000000</td>
      <td>0.303877</td>
      <td>0.512216</td>
      <td>0.262422</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.437849</td>
      <td>0.000000</td>
      <td>0.605204</td>
      <td>0.327616</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.437849</td>
      <td>0.000000</td>
      <td>0.378118</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.658575</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.492771</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.568732</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000000</td>
      <td>0.488122</td>
      <td>0.488122</td>
      <td>0.000000</td>
      <td>0.488122</td>
      <td>0.000000</td>
      <td>0.216677</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.488122</td>
    </tr>
  </tbody>
</table>

The `TfidfVectorize() `will tokenize documents, learn the vocabulary and inverse document frequency weightings, and allow you to encode new documents. Alternately, if you already have a learned `CountVectorize()`.

## Hash Vectorizer

Counts and frequencies can be very useful, but one limitation of these methods is that the vocabulary can become very large. This will require large vectors for encoding documents and impose large requirements on memory and slow down algorithms. A clever work around is to use a one way hash of words to convert them to integers. The clever part is that no vocabulary is required and you can choose an arbitrary-long fixed length vector. A downside is that the hash is a one-way function so there is no way to convert the encoding back to a word (which may not matter for many supervised learning tasks).

The `HashingVectorizer() `class implements this approach that can be used to consistently hash words, then tokenize and encode documents as needed. The example below demonstrates the `HashingVectorizer()`for encoding a single document.

```python
from sklearn.feature_extraction.text import HashingVectorizer
hash = HashingVectorizer(n_features=4)
x = pd.DataFrame(data=hash.fit_transform(df[0]).toarray())
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.894427</td>
      <td>0.447214</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.816497</td>
      <td>0.408248</td>
      <td>0.000000</td>
      <td>0.408248</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.707107</td>
      <td>0.707107</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.894427</td>
      <td>0.447214</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.577350</td>
      <td>0.577350</td>
      <td>0.577350</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>

## Limitations of BoW

- **Vocabulary**: The vocabulary requires careful design, most specifically in order to manage the size, which impacts the sparsity of the document representations.
- **Sparsity**: Sparse representations are harder to model both for computational reasons (space and time complexity) and also for information reasons, where the challenge is for the models to harness so little information in such a large representational space.
- **Meaning**: Discarding word order ignores the context, and in turn meaning of words in the document (semantics). Context and meaning can offer a lot to the model, that if modeled could tell the difference between the same words differently arranged (“this is interesting” vs “is this interesting”), synonyms (“old bike” vs “used bike”), and much more.