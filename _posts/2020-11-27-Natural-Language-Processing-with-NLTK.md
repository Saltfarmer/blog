---
title: "Natural Language Processing with NLTK"
header :
  teaser: /assets/images/sklearn_head.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - NLTK
  - Classification
  - Sklearn

---

**Natural Language Processing** (NLP)  is broadly defined as the automatic manipulation of natural language, like speech and text. Natural language is primarily hard because it is messy. Yet human can easily understand each other most of the time. Document/Text classification is one of the important and typical task in supervised machine learning. Assigning categories to documents, which can be a anything which based from text data.

**Sentiment analysis** is an approach to analyze data and retrieve sentiment that it embodies. Twitter is one of the most common platforms widely used by people to practicing sentiment analysis. The tweet format is very small, which generates a whole new dimension of problems like the use of slang, abbreviations, etc.  

**Natural Language Tool Kit** (NLTK) is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum. It provides a practical introduction to programming for language processing. There are something that we need to do before we classify the text data.

## Installing NLTK (Forgot this part if you already install it)

```python
!pip install nltk

import nltk 
nltk.download() 
```

`nltk.download()` is called to download necessary part of NLTK function. Why not download everything you can ? because NLTK is so huge and had so many features that you can use.

![](https://i.ibb.co/kBrSN2J/2020-11-27-12-40-00-NLTK-Downloader.jpg)

# Text Preprocessing

Text preprocessing simply means to bring your text into a form that is predictable, readable, and analyzable for your task. A task here is a combination of approach and domain. Most common text preprocessing are 

## Tokenization

It is just the term used to describe the process of converting the normal text strings into a list of words (tokens). Sentence tokenizer can be used to find the list of sentences, and Word tokenizer can be used to find the list of words in strings.

```python
text.split()
```

> 'I can fly'
>
> 'I', 'can', 'fly'

## Lowercase

Lowercasing ALL your text data, although commonly overlooked, is one of the simplest and most effective form of text preprocessing. It is applicable to most text mining and NLP problems and can help in cases where your dataset is not very large and significantly helps with consistency of expected output. Lowercasing the text can reduce the size of the vocabulary of our text data.

```python
text.lower()
```

>'I can fly'
>
>'i can fly'

## Punctuation Removal

We remove punctuations so that we don’t have different forms of the same word. If we don’t remove the punctuation, then been. been, been! will be treated separately.

```python
import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)
```

## Stopword Removal

Stop words are a set of commonly used words in a language. Examples of stop words in English are “a”, “the”, “is”, “are” and etc. The intuition behind using stop words is that, by removing low information words from text, we can focus on the important words instead. In my experience, stop word removal, while effective in search and topic extraction systems, showed to be non-critical in classification systems. However, it does help reduce the number of features in consideration which helps keep your models decently sized.

```python
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
```

# Text Vectorization

Currently, we have the messages as lists of tokens (also known as [lemmas](http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)) and now we need to convert each of those messages into a vector the SciKit Learn's algorithm models can work with.

Now we'll convert each message, represented as a list of tokens (lemmas) above, into a vector that machine learning models can understand.

We'll do that in three steps using the bag-of-words model:

1. Count how many times does a word occur in each message (Known as term frequency)
2. Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
3. Normalize the vectors to unit length, to abstract from the original text length (L2 norm)

Let's begin the first step:

Each vector will have as many dimensions as there are unique words in the SMS corpus. We will first use SciKit Learn's **CountVectorizer**. This model will convert a collection of text documents to a matrix of token counts.

We can imagine this as a 2-Dimensional matrix. Where the 1-dimension is the entire vocabulary (1 row per word) and the other dimension are the actual documents, in this case a column per text message. 

```python
from sklearn.feature_extraction.text import CountVectorizer

bag_of_words = CountVectorizer().fit_transform(text)
```

