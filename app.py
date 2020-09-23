import numpy as np
from flask import Flask,request,jsonify,render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
import pickle
import nltk
url = "GoogleNews-vectors-negative300.bin"
embeddings = gensim.models.KeyedVectors.load_word2vec_format(url, binary=True)





app = Flask(__name__,template_folder='templates')
model = pickle.load(open("xgb_reg.pkl", "rb"))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    l = [str(x) for x in request.form.values()]
    print("-----------------------------------------",l)
    stopwords = nltk.corpus.stopwords.words('english')
    l = l[0].lower().replace('[^a-z ]', '')
    l = l.split(' ')
    temp_1 = pd.DataFrame()
    for word in l:
        if word not in stopwords:
            try:
                word_vec = embeddings[word]
                temp_1 = temp_1.append(pd.Series(word_vec), ignore_index = True)
            except:
                pass
        k = temp_1.mean()
    prediction = model.predict(pd.DataFrame(k).T)
    try:
        if(prediction[0] == 1):
            output = "Disaster tweet"
            return render_template('index.html', prediction_text='Tweet actually meant to be! $ {}'.format(output))
        elif(prediction[0] == 1):
            output = "genuine tweet"
            return render_template('index.html', prediction_text='Tweet actually meant to be! $ {}'.format(output))
    except:
        pass
    return render_template('index.html', prediction_text='Tweet actually meant to be! $ {}'.format("genuine tweet")) 

    


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)