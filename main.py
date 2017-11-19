import pip
import nltk
def install(package):
    pip.main(['install', package])

install('gensim')
nltk.download('punkt')
nltk.download('stopwords')

import gensim
import string
import numpy as np
import pandas as pd
from gensim import corpora
from pymongo import MongoClient
from bson.objectid import ObjectId
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    doc= str(doc)
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

documents= pd.read_csv('KO_raw_withBODY.csv')['body']

print("Cleaning Data...")
documents_clean = [clean(KO_doc).split() for KO_doc in documents]

print("Construct Document matrix...")
dict_doc = corpora.Dictionary(documents_clean)
documents_term_matrix = [dict_doc.doc2bow(doc) for doc in documents_clean]

Lda_KO= gensim.models.LdaMulticore
print ("Training Model...")
ldamodel_KO= Lda_KO(documents_term_matrix , num_topics=500, id2word=dict_doc, passes=50,workers=3)
ldamodel_KO.save(fname="ldaModel_ko_500pass")

ldamodel_KO.print_topics(num_topics=3,num_words=20)

Topics_predictions= np.array(ldamodel_KO.get_document_topics(documents_term_matrix))
