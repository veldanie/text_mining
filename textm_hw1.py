
#import packages
import numpy as np
import matplotlib.pyplot as plt
import nltk as nl
from nltk.tokenize import word_tokenize
import pandas as pd

#Import state-of-the-union speech
text_raw = pd.read_csv('./text-mining-tutorial/speech_data_extend.txt', sep='\t')

#Consider paragraphs after 2000.
text_data = text_raw.loc[text_raw['year']>=2000, :]

#Number of paragraphs:
print(len(text_data))

#########################################################################
#########################################################################
#########################################################################
##1. Preprocessing of the data
from stop_words import get_stop_words
stop_words = get_stop_words('en')
from nltk.stem.porter import PorterStemmer
st = PorterStemmer()
docs = pd.Series(np.zeros(text_data.shape[0]))
tokens = [] #All the list of words.

for i, line in enumerate(text_data['speech']):
    #Tokenize the data:
    doc_i = word_tokenize(line.lower())
    #Remove non-alphabetic characters:
    doc_i = [tok for tok in doc_i if tok.isalpha()]
    #Remove stopwords using a list of your choice:
    doc_i = [tok for tok in doc_i if tok not in stop_words]
    #Stem the data using the Porter stemmer:
    doc_i = [st.stem(tok) for tok in doc_i]

    tokens.extend(doc_i)
    docs.iloc[i] = doc_i


# Corpus-level tf-idf score for every term, and choose a cutoff below which to remove words.
unique_words = np.unique(tokens)
lw = len(unique_words) # Number of words
ld = len(docs) # Number of documents

word_count = nl.FreqDist(tokens)
tf = {k: 1+np.log(v) for k, v in word_count.items()}
df = {k: np.sum(list(map(lambda x: k in x, docs))) for k in word_count.keys()}
idf = {k: np.log(ld/v) for k, v in df.items()}
rank = {k: v*u for k, v, u in zip(tf.keys(), tf.values(), idf.values())}

# Based on the ranking we select aprox. 500 words.
selected_words = {k: v for k, v in rank.items() if v>16.5}
ls = len(selected_words) # Length of selected words.

#Document-term matrix using aprox. 500 words selected using the tf-idf score.
X = pd.DataFrame(np.zeros(shape = (ld, ls)), columns = selected_words.keys())

for w in selected_words.keys():
    X[w] = list(map(lambda x: x.count(w), docs))

#########################################################################
#########################################################################
#########################################################################

##2. Analysis:


#########################################################################
#########################################################################
#########################################################################


##3. Generate the tf-idf-weighted document-term matrix S. Perform SVD.

tf = X.copy()
tf[tf > 0] = 1+ np.log(X[X>0])

S = X.copy()
for i in range(ls):
    S[tf.columns[i]] = tf.iloc[:,i] * idf[tf.columns[i]]

# Function that estimates Cosine similarity:
def cos_sim (di, dj):
    sim = np.dot(di,dj)/(np.sqrt(np.dot(di,di))*np.sqrt(np.dot(dj,dj)))
    return (sim)

# Singular Value Decomposition:
S_svd = np.linalg.svd(S, full_matrices=1, compute_uv=1)


#########################################################################
#########################################################################
#########################################################################

#4. Multinomial Mixture Model using EM algorithm:

#First, we define the log-likelihood:
def log_lik (X, B, rho):
    ll = np.sum(list(map(lambda x: np.log(np.sum([rho_i*np.prod(B_i**x) for rho_i, B_i in zip(rho, B)])), np.array(X))))
    return(ll)

#E-M Algorithm:
K=2 #Number of Topics
B = np.random.dirichlet(np.ones(ls), K) #Initial Beta matrix.
rho = np.ones(K)/K # Initial rho vector.
max_iter = 100 # Max number of iterations
ll = [log_lik(X,B,rho)]
for i in range(max_iter):
    #E-step (lecture 3, slide 19):
    z = np.array(list(map(lambda x: [rho_i*np.prod(B_i**x) for rho_i, B_i in zip(rho, B)], np.array(X))))
    for i in range(ld):
        z[i] = z[i]/np.sum(z[i])

    #M-step (lecture 3, slide 20):
    rho = np.sum(z, axis = 0) / np.sum(np.sum(z, axis = 0))
    for k in range(K):
        B[k] = np.sum([z_i*x_i for z_i, x_i in zip(z[:,k], np.array(X))], axis = 0) / np.sum([z_i*x_i for z_i, x_i in zip(z[:,k], np.sum(np.array(X),axis = 1))])

    #log-likelihood at each iteration:
    ll.extend([log_lik(X,B,rho)])
    delta = np.abs(ll[-2] - ll[-1])
    if delta < 1:
        break

## Top terms per topic:
top_terms = []
for k in range(K):
    new_terms = X.columns[B[k]>0.006]
    top_terms.append(new_terms)

plt.plot(ll)
plt.xlabel('Iterations');plt.ylabel('log-likelihood')
plt.title('log-likelihood')
plt.show()
