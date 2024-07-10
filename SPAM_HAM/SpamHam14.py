#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing- NLP
# 
# NLP is a field concerned with the ability of a computer to understand, analyse, manipulate and potentially generate human language.
# NLP is a broad umbrella that encompasses many topics. Some of them are sentiment analysis, topic modelling, text classification etc
# NLTK- Natural Language Toolkit- The NLTK is the most utilised package for handling natural language processing tasks. It is an open source library.

# # Spam/Ham classificatation using NLP

# In[2]:


import nltk
import pandas as pd
import numpy as np


# In[3]:


pip install NLTK


# In[4]:


dataset= pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)

dataset.columns=['label','body_text']

dataset.head()


# In[5]:


dataset['body_text'][1]


# In[6]:


dataset['body_text'][0]


# In[7]:


#What is the shape of the data

print("Input data has {} rows and {} columns".format(len(dataset),len(dataset.columns)))


# In[8]:


print("Out of {} rows, {} are spam. {} are ham".format(len(dataset),len(dataset[dataset['label']=='spam']),
                                                      len(dataset[dataset['label']=='ham'])))


# In[9]:


#Missing data calculate 

print("Number of null in label: {}".format(dataset['label'].isnull().sum()))
print("Number of null in text: {}".format(dataset['body_text'].isnull().sum()))


# Preprocessing text data- Cleaning up the text data is necessary to highlight attributes that you are going to use in ML algorithms.
# Cleaning or preprocessing the data consists of a number of steps
# -
# -Remove Punctuation
# 
# -Tokenization
# 
# -Remove Stopwords
# 
# -Lemmatize/Stemming

# In[10]:


import string
string.punctuation


# In[11]:


def remove_punct(text):
    text_nonpunct="".join([char for char in text if char not in string.punctuation])
    return text_nonpunct

dataset['body_text_clean']=dataset['body_text'].apply(lambda x:remove_punct(x))
dataset.head()


# In[12]:


#Tokenization- Tokenizing is splitting some string or sentence into a list of words

import re

def tokenize(text):    
    tokens=re.split('\W',text)    
    return tokens

dataset['body_text_tokenized']=dataset['body_text_clean'].apply(lambda x:tokenize(x.lower()))

dataset.head()


# In[13]:


#Remove Stopwords- These are comonly used words like the, and, but, if that don't contribut much to the meaning of a sentence.

import nltk

# Download the stopwords from NLTK
nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords = stopwords.words('english')

def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopwords]
    return text

dataset['body_text_nostop'] = dataset['body_text_tokenized'].apply(lambda x: remove_stopwords(x))
dataset.head()


# Stemming- Stemming is the process of reducing inflected or derived words to their stem or root.
# 
# 
# Lemmatization- It is the process of grouping together the inflected forms of a word so they can be analysed as a single term, identified by the word's lemma.
# For e.g. type, typing and typed are forms of the same lemma type.

# In[14]:


import nltk

wn=nltk.WordNetLemmatizer()

def lemmatizing(tokenized_text):
    text=[wn.lemmatize(word) for word in tokenized_text]
    return text

dataset['body_text_lemmatized']=dataset['body_text_nostop'].apply(lambda x:lemmatizing(x))

dataset.head()


# # Vectorization- This is defined as the process of encoding text as integers to create feature vectors. In out ontext we will be taking inividual text messages and converting it to a numeric vector that represents that text message.
# Count Vectorization- This creates a document-term matrix where the entry of each cell will be a count of the number of times that word occured in that document.

# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

# Assuming the dataset is already loaded in a DataFrame called dataset
# dataset = pd.read_csv('your_dataset.csv')

# Initialize the PorterStemmer

ps = PorterStemmer()

# Define stopwords
stopwords = set(stopwords.words('english'))

# Define the text cleaning function
def clean_text(text):
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

# Initialize CountVectorizer with the custom analyzer
count_vect = CountVectorizer(analyzer=clean_text)

# Fit and transform the text data
X_count = count_vect.fit_transform(dataset['body_text'])

# Print the shape of the resulting sparse matrix
print(X_count.shape)


# #Apply count vectorizer to a smaller sample
# Sparse Matrix- A matrix in which most entries are 0. In the interest of efficient storage, a sparse matrix will be stored by only storing the lcoations of the non-zero elements.

# In[16]:


data_sample=dataset[0:20]

count_vect_sample= CountVectorizer(analyzer=clean_text)
X_count_sample= count_vect_sample.fit_transform(data_sample['body_text'])

print(X_count_sample.shape)


# In[17]:


X_count_sample


# In[18]:


x_counts_df= pd.DataFrame(X_count_sample.toarray())
x_counts_df


# import warnings
# warnings.filterwarnings("ignore")
# 
# x_counts_df.columns= count_vect_sample.get_feature_names()
# x_counts_df

# # TF-IDF (Term Frequency, Inverse DOcument Frequency)- Creates a document term amtrix where the column represents single unique terms(unirams) but the cell represents a weighting meant to represent how important a word is to a document.

# In[19]:


import warnings 
warnings.filterwarnings("ignore")

x_counts_df.columns= count_vect_sample.get_feature_names() 
x_counts_df


# # TF-IDF (Term Frequency, Inverse Document Frequency)- Creates a document term matrix where the column represents single unique terms(unirams) but the cell represents a weighting meant to represent how important a word is to a document.

# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect=TfidfVectorizer(analyzer=clean_text)
X_tfidf= tfidf_vect.fit_transform(dataset['body_text'])

print(X_tfidf.shape)


# In[21]:


#Apply TfidfVectorizer to a smaller sample

data_sample=dataset[0:20]
tfidf_vect_sample=TfidfVectorizer(analyzer=clean_text)
X_tfidf_sample=tfidf_vect_sample.fit_transform(data_sample['body_text'])
print(X_tfidf_sample.shape)


# In[22]:


x_tfidf_df= pd.DataFrame(X_tfidf_sample.toarray())
x_tfidf_df.columns=tfidf_vect_sample.get_feature_names()
x_tfidf_df


# In[23]:


#Feature Engineering: Feature Creation


dataset=pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
dataset.columns=['label','body_text']
dataset.head()


# In[24]:


#Create feature for text message length

dataset['body_len']=dataset["body_text"].apply(lambda x:len(x)-x.count(" "))
dataset.head()


# In[25]:


#create feature for % of text that is punctuation

def count_punct(text):    
    count=sum([1 for char in text if char in string.punctuation])    
    return round(count/(len(text)-text.count(" ")),3)*100

dataset['punct%']=dataset['body_text'].apply(lambda x:count_punct(x))
dataset.head()


# In[26]:


import matplotlib.pyplot as plt
import numpy as np

bins=np.linspace(0,200,40)

plt.hist(dataset['body_len'],bins)
plt.title('Body Length Distribution')
plt.show()


# In[27]:


bins=np.linspace(0,50,40)

plt.hist(dataset['punct%'],bins)
plt.title('Punctuation % Distribution')
plt.show()


# In[28]:


def count_punct(text):
    count=sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text)-text.count(" ")),3)*100

dataset['body_len']=dataset["body_text"].apply(lambda x:len(x)-x.count(" "))
dataset['punct%']=dataset['body_text'].apply(lambda x:count_punct(x))

dataset.head()


# In[30]:


X_features=pd.concat([dataset['body_len'],dataset['punct%'],pd.DataFrame(X_tfidf.toarray())],axis=1)
X_features.head()


# # Model using K-Fold cross validation
# Cross Validation -
# Div the dataset into K-multiple folds and train in one of the folds and test  at n-1 fold;  validating at last fold
# 
# Dividing data into 5 folds - last one fold for validation --initially; then repeated for all folds(iterative )
# then repeated for all the folds 
# 
# So entire datset for training as well as for testing 

# In[31]:


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score

rf=RandomForestClassifier(n_jobs=1)
k_fold=KFold(n_splits=5)

cross_val_score(rf, X_features, dataset['label'],cv=k_fold, scoring='accuracy',n_jobs=1)


# # Model Using Train Test Split

# In[32]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, y_train, y_test= train_test_split(X_features, dataset['label'],test_size=0.3,random_state=0)


# In[34]:


rf=RandomForestClassifier(n_estimators=500, max_depth=20, n_jobs=-1)
rf_model=rf.fit(X_train, y_train)


# In[35]:


sorted(zip(rf_model.feature_importances_,X_train.columns),reverse=True)[0:10]


# In[37]:


y_pred= rf_model.predict(X_test)

precision, recall, fscore, support=score(y_test, y_pred, pos_label='spam',average='binary')


# In[38]:


print('Precision {} / Recall {} / Accuracy {}'.format(round(precision,3),
                                                     round(recall,3),
                                                     round((y_pred==y_test).sum()/len(y_pred),3)))


# In[ ]:




