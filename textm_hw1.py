
#import packages
import numpy as np
import matplotlib.pyplot as plt
import nltk as nl
from nltk.tokenize import word_tokenize
import pandas as pd

#Import state-of-the-union speech
text_raw = pd.read_csv('./speech_data_extend.txt', sep='\t')


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
tokens = [] #List of all words.
# Download corpora if necessary: nl.download()


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
tfidf = {k : v * tf[k] for k, v in idf.items() if k in tf}


# Based on the ranking we select 500 words with highest tf-idf
# 1st we get the rank
import operator
rank = sorted(tfidf.items(), key=operator.itemgetter(1), reverse=True)
cutoff = rank[500][1]
# 2nd apply the cut-off
selected_words = {k: v for k, v in tfidf.items() if v>cutoff}
ls = len(selected_words) # number of selected words: 500

#Document-term matrix using 500 words selected using the tf-idf score.
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
tf[tf > 0] = 1+ np.log(X[X>0]) #term frequency for each word and each document.

S = X.copy()
for i in range(ls):
    S[tf.columns[i]] = tf.iloc[:,i] * idf[tf.columns[i]] #tf*inverse document frequency

# Singular Value Decomposition:
S_svd = np.linalg.svd(S, full_matrices=1, compute_uv=1)
#X = A SIGMA B
A = S_svd[0]
SIGMA = np.vstack((np.diag(S_svd[1]),np.zeros(shape = (ld-ls, ls))))
B = S_svd[2]

#We retain 200 singular values and approximate S.
SIGMA2 = SIGMA.copy()
for i in range(200,ls-1):
    SIGMA2[i,i] = 0

S_hat = A.dot(SIGMA2).dot(B)


np.sum(text_data['president']=='Obama')
np.sum(text_data['president']=='BushII')

# Function that estimates Cosine similarity:
def cos_sim (di, dj):
    if np.sum(di)==0 or np.sum(dj)==0:
        sim = 0
    else:
        sim = np.dot(di,dj)/(np.sqrt(np.dot(di,di))*np.sqrt(np.dot(dj,dj)))
    return (sim)

#Cosine similarites using S and S_hat:
S_B = np.array(S)[text_data['president']=='BushII']
S_O = np.array(S)[text_data['president']=='Obama']

S_hat_B = S_hat[text_data['president']=='BushII']
S_hat_O = S_hat[text_data['president']=='Obama']

#Bush within Average Cosine Similarity:
bb1 = np.mean([cos_sim(S_B[i],S_B[j]) for i in range(S_B.shape[0]) for j in range(S_B.shape[0])])
bb2 = np.mean([cos_sim(S_hat_B[i],S_hat_B[j]) for i in range(S_hat_B.shape[0]) for j in range(S_hat_B.shape[0])])

#Obama within Average Cosine Similarity:
oo1 = np.mean([cos_sim(S_O[i],S_O[j]) for i in range(S_O.shape[0]) for j in range(S_O.shape[0])])
oo2 = np.mean([cos_sim(S_hat_O[i],S_hat_O[j]) for i in range(S_hat_O.shape[0]) for j in range(S_hat_O.shape[0])])

#Bush-Obama cross Average Cosine Similarity:
bo1 = np.mean([cos_sim(S_B[i],S_O[j]) for i in range(S_B.shape[0]) for j in range(S_O.shape[0])])
bo2 = np.mean([cos_sim(S_hat_B[i],S_hat_O[j]) for i in range(S_hat_B.shape[0]) for j in range(S_hat_O.shape[0])])

ind = np.arange(3)  # the x locations for the groups
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(ind, (bb1, oo1, bo1), width, color='r')
rects2 = ax.bar(ind + width, (bb2, oo2, bo2), width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('Average Cosine Similarity')
ax.set_title('Average Cosine Similarity \n within and across Bush and Obama (2000, 2014)')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Bush-Bush', 'Obama-Obama', 'Bush-Obama'))
ax.legend((rects1[0], rects2[0]), ('S', 'S_hat'))

plt.show()

fig.savefig('cs.png')
##################################################################################
##################################################################################
##################################################################################

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
    new_terms = X.columns[B[k]>0.0055]
    top_terms.append(new_terms)

fig, ax = plt.subplots()
plt.plot(ll)
plt.xlabel('Iterations');plt.ylabel('Log-likelihood')
plt.title('Log-likelihood Function')
plt.show()

fig.savefig('ll.png')
