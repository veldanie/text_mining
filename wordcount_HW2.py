
import os
os.getcwd()
os.chdir('/Users/sauravpoudel/Desktop/Serious Text')

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize

text_raw = pd.read_csv('./speech_data_extend.txt', sep='\t')

type(text_raw)
text_raw.shape
text_raw.head()

#Consider paragraphs after 1946.
text_data = text_raw.loc[text_raw['year']>=1946, :]

# last to observations excluded for test data
train_data = text_data.iloc[:-10,:]
len(train_data)
test_data = text_data.iloc[len(train_data): ,:]
len(test_data)



frames = [train_data, test_data]
join_data = pd.concat(frames)




corpus = []

##1. Preprocessing of the data
from stop_words import get_stop_words
stop_words = get_stop_words('en')
from nltk.stem.porter import PorterStemmer
st = PorterStemmer()

for i, line in enumerate(join_data['speech']):
    
    #Tokenize the data:
    doc = word_tokenize(line.lower())
    #Remove non-alphabetic characters:
    doc = [tok for tok in doc if tok.isalpha()]
    #Remove stopwords using a list of your choice:
    doc = [tok for tok in doc if tok not in stop_words]
    #Stem the data using the Porter stemmer:
    doc = [st.stem(tok) for tok in doc]

    corpus.append(doc)


result = []
for i in range(0,len(corpus)):
    str1 = ' '.join(corpus[i])
    result.append(str1)
    
# Count Vectorizer used for words per document
from sklearn.feature_extraction.text import CountVectorizer   

vectorizer = CountVectorizer(analyzer = 'word',tokenizer = word_tokenize,lowercase = True,stop_words = 'english',max_features=5000)



X = vectorizer.fit_transform(result)
feature_names = vectorizer.get_feature_names()
dense = X.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

df.head
df.shape
len(df)
type(df)


# Splitting this matrix into train and test again: 

train_df = df.iloc[:len(train_data),:]
test_df = df.iloc[len(train_data): ,:]

len(train_df)
len(test_df)


#(The presidents in the sample are: Truman (D), 
#Eisenhower (R), Kennedy (D), Johnson (D), Nixon (R), 
#Ford (R), Carter (D), Reagan (R), Bush Senior (R), 
#Clinton (D), Bush Jr (R), and Obama (D)).

president = pd.DataFrame(np.zeros((len(train_df),1)))
president.columns = ["President"]


#Democrat = 1, Republican = 0




for i in range(len(train_df)):
    if text_data.president.iloc[i] in ['Truman','Kennedy','Johnson','Carter','Clinton','Obama']:
        president.iloc[i][0] = 1
    else:
        president.iloc[i][0] = 0
    

# checking lengths of all till now: 

len(president)
len(train_df)
len(test_df)
president.shape
train_df.shape
test_df.shape


from sklearn.linear_model import LogisticRegression

    
log_model = LogisticRegression(penalty='l1', solver='liblinear', C=1)
log_model = log_model.fit(X=train_df, y=president.values.ravel())


y_pred = log_model.predict(test_df)
print(y_pred)
# Checking with orignal test lables
test_data.president

# Model selection is not availabel in my version of python. 

#from sklearn.model_selection import GridSearchCV


# prepare a range of C values to test
#C = np.array([1,0.75,0.5,0.25,0.01,0])

#model = LogisticRegression(penalty='l1', solver='liblinear')
#grid = GridSearchCV(estimator=model, param_grid=dict(C=C))
#grid.fit(train_df, president.values.ravel())

# summarize the results of the grid search
#print(grid.best_score_)
#print(grid.best_estimator_.alpha)


