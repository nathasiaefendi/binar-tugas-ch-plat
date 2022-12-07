import sqlite3
import pandas as pd
import re,string
import seaborn as sns
import scipy.stats.distributions as dist
import sqlite3

from flask import Flask, request, jsonify, send_file
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
from cleantext import clean
from matplotlib import pyplot as plt
from functools import reduce
from cgitb import text
from datetime import datetime

import pickle

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from collections import defaultdict

import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, SimpleRNN, Activation
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras import backend as K

# from sklearn.model_selection import train_test_split
# from sklearn import metrics

from keras.models import load_model


app = Flask(__name__)
app.json_encoder = LazyJSONEncoder

swagger_template = dict(
    info = {
        'title': LazyString(lambda: 'API Documentation for Deep Learning'),
        'version' : LazyString(lambda : '1.0.0'),
        'description' : LazyString(lambda : 'Dokumentasi API untuk Deep Learning'),
    },
    host = LazyString(lambda : request.host)
)

swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint': 'docs',
            'route': '/docs.json',
        }
    ],
    'static_url_path': "/flasgger_static",
    'swagger_ui': True,
    'specs_route': "/"
}

swagger = Swagger(app, template=swagger_template, config = swagger_config)

#base param
max_features = 10000
sentiment = ['negative', 'neural', 'positive']

#pipeline function untuk memudahkan penggunaan function
def pipeline_function(data, fns):
    return reduce(lambda a, x: x(a),
                  fns,
                  data)

#text file helper : slang & stopwords
#import data slang
slang_list = pd.read_csv('file-helper/new_kamusalay.csv', delimiter = ",", encoding='latin-1', header=None, names=["slang","meaning"])
slang_list['slang'] = slang_list['slang'].str.lower()
slang_list['meaning'] = slang_list['meaning'].str.lower()

#import data stopwords
stopword_list = open('file-helper/stopwords.txt','r').read().splitlines()
stopword_list = list(map(str.lower,stopword_list))

#import data abusive word
abusive_list = pd.read_csv('file-helper/abusive.csv', delimiter = ",", encoding='latin-1')
abusive_list = abusive_list['ABUSIVE'].values
abusive_list = list(map(str.lower,abusive_list))


###Text cleansing function
#text cleansing based on twitter hatespeech. It cleanse links, retweet, user, mentions, hashtags, emoji, resulting only the alphabet on the text

#cleansing functions
#strip links/hyperlink(if exist) 
def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

#strip mentions and hashtags
def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

#strip retweet
def remove_rt_user_url(text):
    text = text.replace('rt', '')
    text = text.replace('user', '')
    text = text.replace('url', '')
    return text

#remove emoji
def remove_emoji(text):
    return clean(text, no_emoji=True)

#remove other things other than what has been cleansed by functions above
def alphabet_only(text):
    return re.sub(r'[^a-zA-Z]', ' ', text)


##Load model
file = open('resources/x_pad_sequences.pickle', 'rb')
feature_pickle = pickle.load(file)
file.close()

model_cnn = load_model('model/cnn/model.h5')
model_rnn = load_model('model/rnn/model.h5')
model_lstm = load_model('model/lstm/model.h5')

##Advanced cleansing:
#remove unnecesary word, stopwords and slang to get only the meaningful word based on the text
def replace_slang(text):
    text_list = text.split()
    cleansed_text = []
   
    for word in text_list:
      used_word = word
      for i, slang in enumerate(slang_list.slang):
        if slang == word:
          used_word = slang_list.meaning[i]
      cleansed_text.append(used_word)
    return ' '.join(cleansed_text)

def replace_stopwords(text):
    word_list = text.split()
    cleansed_text = [word for word in word_list if word not in stopword_list]
    return ' '.join(cleansed_text)

def check_abusive_word(text):
    has_abusive = 0
    for abusive in abusive_list:
      if abusive in text:
        has_abusive = 1
        break
    return has_abusive


def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text


def clean_text(text):
    text = str(text).lower()
    return pipeline_function(text, [
                     remove_rt_user_url,
                     remove_emoji,
                     strip_links,
                     strip_all_entities,
                     alphabet_only,
                     replace_slang,
                     replace_stopwords])

#database function
def save_to_sqllite(raw, output, sentiment = ''):
    try:
        conn = sqlite3.connect('text_cleansing.db')
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS text_cleansing (raw varchar(255) NOT NULL, basic_output VARCHAR(255) NOT NULL, sentiment VARCHAR(255))')

        if(sentiment == ''):
            cursor.execute("INSERT INTO text_cleansing (raw, basic_output) VALUES (?, ?)", [raw, output])
        else:
            cursor.execute("INSERT INTO text_cleansing (raw,basic_output,sentiment) VALUES (?, ?, ?)", [raw, output, sentiment])

        conn.commit()
        cursor.close()
        conn.close()
        return True
    except:
        return False
    
#Dataframe text cleansing
def df_text_cleansing(df):
    tweet_clean = df.tweet.str.lower()
    tweet_clean = tweet_clean.apply(clean_text)
    df['tweet_clean']  = tweet_clean
    return df

#JSON Balikan text cleansing
# text cleansing
def text_cleansing_success(text, cleansed_text):
    return {
            'status_code': 200,
            'description': 'Sukses membersihkan text',
            'before' : text,
            'after': cleansed_text,
    }     

def text_cleansing_error():
    return{
            'status_code': 500,
            'description': 'Error menyimpan ke database',
    }   

#text cleansing file
def text_cleansing_file_error_column(columnLength):
    return {
         'status_code' : 500,
         'description' : 'File tidak sesuai {0} column. Harusnya 13'.format(columnLength),
    }

def text_cleansing_file_error():
    return {
            'status_code' : 500,
            'description' : 'File error',
    }


def get_feature(original_text):
    text = [clean_text(original_text)]
    tokenizer = Tokenizer(num_words= max_features, split = ' ', lower = True)
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen = feature_pickle.shape[1])
    return feature

def predict_lstm(feature):
    prediction = model_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    return get_sentiment

def get_sentiment_lstm(text):
    return predict_lstm(get_feature(text))

def predict_rnn(feature):
    prediction = model_rnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    return get_sentiment

def get_sentiment_rnn(text):
    return predict_rnn(get_feature(text))

def predict_cnn(feature):
    prediction = model_cnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    return get_sentiment

def get_sentiment_cnn(text):
    return predict_cnn(get_feature(text))

#SWAGGER URL
#cnn model
@swag_from('docs/CNN.yml', methods=['POST'])
@app.route('/cnn', methods=['POST'])
def cnn():
    original_text = request.form.get('text')
    feature = get_feature(original_text)

    prediction = model_cnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': 'Result of Sentiment Analysis using CNN',
        'data': {
            'text': original_text,
            'sentiment': get_sentiment,
        }
    }

    response_data = jsonify(json_response)
    return response_data

#rnn model
@swag_from('docs/RNN.yml', methods=['POST'])
@app.route('/rnn', methods=['POST'])
def rnn():
    original_text = request.form.get('text')
    feature = get_feature(original_text)

    prediction = model_rnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': 'Result of Sentiment Analysis using RNN',
        'data': {
            'text': original_text,
            'sentiment': get_sentiment,
        }
    }

    response_data = jsonify(json_response)
    return response_data

#lstm model
@swag_from('docs/LSTM.yml', methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    get_sentiment = predict_lstm(get_feature(original_text))

    json_response = {
        'status_code': 200,
        'description': 'Result of Sentiment Analysis using LSTM',
        'data': {
            'text': original_text,
            'sentiment': get_sentiment,
        }
    }

    response_data = jsonify(json_response)
    return response_data

@app.route('/')
def home():
    return "Hello World!"

#file rnn
@swag_from('docs/file_predict.yml', methods=['POST'])
@app.route('/file_predict_rnn', methods=['POST'])
def predict_file_rnn():
    """Cleanse Text & Predit model (based on RNN) """
     #get file
    file = request.files["file"] 

     #cek apakah tipe file sesuai         
    if file and file.content_type  == 'application/vnd.ms-excel':     
        #baca dataframe
        df = pd.read_csv(file, encoding='latin-1')
        df.columns = map(str.lower, df.columns)

        #cek apakah kolom sesuai
        if(len(df.columns) != 13):
              return jsonify(text_cleansing_file_error_column(len(df.columns)))
        else:
            #cleansing text 20 random. karena berat
            df = df_text_cleansing(df.sample(n=20, random_state=1))
            df['tweet_clean_advanced'] = df['tweet_clean'].apply(clean_text)
            data_output = df[['tweet', 'tweet_clean', 'tweet_clean_advanced']]

            # save to sql
            for i, tweet_clean in enumerate(data_output['tweet_clean']): print(data_output.iloc[[i]]['tweet'], tweet_clean, data_output.iloc[[i]]['tweet_clean_advanced'])
            #save ke csv
            file_name = 'file_cleansing_advanced_{0}.csv'.format(datetime.now().strftime('%m%d%Y_%H%M%S'))
            data_output.to_csv('result/' + file_name)

            return send_file(
                'result/' + file_name,
                as_attachment = True,
                download_name= file_name,
            )
    else:        return jsonify(text_cleansing_file_error())

#file rnn
@swag_from('docs/file_predict.yml', methods=['POST'])
@app.route('/file_predict_cnn', methods=['POST'])
def predict_file_cnn():
     #get file
    file = request.files["file"] 

     #cek apakah tipe file sesuai         
    if file and file.content_type  == 'application/vnd.ms-excel':     

        #baca dataframe
        df = pd.read_csv(file, encoding='latin-1')
        df.columns = map(str.lower, df.columns)

        #cek apakah kolom sesuai
        if(len(df.columns) != 13):
              return jsonify(text_cleansing_file_error_column(len(df.columns)))
        else:
            #cleansing text 20 random. karena berat
            df = df_text_cleansing(df.sample(n=20, random_state=1))
            df['tweet_clean'] = df['tweet'].apply(clean_text)
            df['sentiment'] = df['tweet_clean'].apply(get_sentiment_cnn)
            data_output = df[['tweet', 'tweet_clean', 'sentiment']]

            # save to sql
            # for i, tweet_clean in enumerate(data_output['tweet_clean']): print(data_output.iloc[[i]]['tweet'], tweet_clean, data_output.iloc[[i]]['sentiment'])
            for i, tweet_clean in enumerate(data_output['tweet_clean']): save_to_sqllite(data_output[i]['tweet'], tweet_clean, data_output[i]['sentiment'])

            #generate json
            json_result = data_output.to_json(orient='records', lines=True)
           
            json_response = {
                'status_code': 200,
                'description': 'Result of Sentiment Analysis using RNN (File Upload)',
                'data': {
                    json_result
                }
            }
            return jsonify(json_response)

    else:        return jsonify(text_cleansing_file_error())

#file rnn
@swag_from('docs/file_predict.yml', methods=['POST'])
@app.route('/file_predict_rnn', methods=['POST'])
def predict_file_rnn():
     #get file
    file = request.files["file"] 

     #cek apakah tipe file sesuai         
    if file and file.content_type  == 'application/vnd.ms-excel':     

        #baca dataframe
        df = pd.read_csv(file, encoding='latin-1')
        df.columns = map(str.lower, df.columns)

        #cek apakah kolom sesuai
        if(len(df.columns) != 13):
              return jsonify(text_cleansing_file_error_column(len(df.columns)))
        else:
            #cleansing text 20 random. karena berat
            df = df_text_cleansing(df.sample(n=20, random_state=1))
            df['tweet_clean'] = df['tweet'].apply(clean_text)
            df['sentiment'] = df['tweet_clean'].apply(get_sentiment_rnn)
            data_output = df[['tweet', 'tweet_clean', 'sentiment']]

            # save to sql
            for i, tweet_clean in enumerate(data_output['tweet_clean']): save_to_sqllite(data_output[i]['tweet'], tweet_clean, data_output[i]['sentiment'])

            #generate json
            json_result = data_output.to_json(orient='records', lines=True)
           
            json_response = {
                'status_code': 200,
                'description': 'Result of Sentiment Analysis using RNN (File Upload)',
                'data': {
                    json_result
                }
            }
            return jsonify(json_response)

    else:        return jsonify(text_cleansing_file_error())

#file lstm
@swag_from('docs/file_predict.yml', methods=['POST'])
@app.route('/file_predict_lstm', methods=['POST'])
def predict_file_lstm():
     #get file
    file = request.files["file"] 

     #cek apakah tipe file sesuai         
    if file and file.content_type  == 'application/vnd.ms-excel':     

        #baca dataframe
        df = pd.read_csv(file, encoding='latin-1')
        df.columns = map(str.lower, df.columns)

        #cek apakah kolom sesuai
        if(len(df.columns) != 13):
              return jsonify(text_cleansing_file_error_column(len(df.columns)))
        else:
            #cleansing text 20 random. karena berat
            df = df_text_cleansing(df.sample(n=20, random_state=1))
            df['tweet_clean'] = df['tweet'].apply(clean_text)
            df['sentiment'] = df['tweet_clean'].apply(get_sentiment_lstm)
            data_output = df[['tweet', 'tweet_clean', 'sentiment']]

            # save to sql
            for i, tweet_clean in enumerate(data_output['tweet_clean']): save_to_sqllite(data_output[i]['tweet'], tweet_clean, data_output[i]['sentiment'])

            #generate json
            json_result = data_output.to_json(orient='records', lines=True)
           
            json_response = {
                'status_code': 200,
                'description': 'Result of Sentiment Analysis using LSTM (File Upload)',
                'data': {
                    json_result
                }
            }
            return jsonify(json_response)

    else:        return jsonify(text_cleansing_file_error())

#text cleansing 
@swag_from('docs/file_predict.yml', methods=['POST'])
@app.route('/file_predict_lstm', methods=['POST'])
def text_cleansing():
    """Cleanse Text & Save it to sqllite"""
    #clean text
    text = request.form.get('text')
    cleansed_text = clean_text(text)
     #save to sql
    db = save_to_sqllite(text, cleansed_text, '')

    #generate balikan json
    json_response = object()
    if(db == True):
        json_response = text_cleansing_success(text, cleansed_text)
    else:
        json_response = text_cleansing_error()

    return jsonify(json_response)




if __name__ == "__main__":
    # Development only: run "python main.py" and open http://localhost:8080
    # When deploying to Cloud Run, a production-grade WSGI HTTP server,
    # such as Gunicorn, will serve the app.
    app.run(host="127.0.0.2", port=8080, debug=True)