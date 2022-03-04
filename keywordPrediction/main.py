from flask import Flask, request
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk as nlp
import re
from scipy.sparse import coo_matrix
import nltk
from flask import send_file
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

username = ''


@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    jsonData = request.data
    print(jsonData)
    return "Hello World!"


@app.route('/data', methods=['GET', 'POST'])
def testd():
    jsonData = request.args.get('text')
    print(jsonData)
    test = getkeywords(jsonData)
    return test


@app.route('/keywords', methods=['GET', 'POST'])
def keywordsApi():
    jsondata = request.args.get('text')
    global username
    username = ''
    username = request.args.get('user')
    print(jsondata)
    print("Username " + username)
    test = getkeywords(jsondata)
    return test


@app.route('/get_image')
def get_image():
    filename = username + "_word.png"
    return send_file(filename, mimetype='image/gif')


@app.route('/getimageuser')
def get_imageuser():
    username = request.args.get("user")
    print(username)
    filename = username + "_word.png"
    return send_file(filename, mimetype='image/gif')


def getkeywords(text):
    # nltk.download('omw-1.4')
    # print("in function")
    text = re.sub("(\W)", " ", text)
    tokenizer = nlp.WordPunctTokenizer()
    word_count = len(tokenizer.tokenize(text))
    # print(word_count)
    freq = pd.Series(text.split()).value_counts()
    # print(freq.head(10))
    # print(freq.tail(10))
    # print(len(freq))
    fileEnglish = "contents/stop_words_english.txt"
    filewords = open(fileEnglish, 'r', encoding="utf8")
    newStopWords = filewords.read()
    newStopWordsList = set(newStopWords.splitlines())
    # print(len(newStopWordsList))
    # print('said' in newStopWordsList)
    nltk.download('wordnet')
    lemma = nlp.WordNetLemmatizer()
    text = lemma.lemmatize(text)
    text = text.lower()
    nltk.download('stopwords')
    stopword_list = set(stopwords.words("english"))
    # print(stopword_list)
    stopword_list = newStopWordsList
    # print('said' in stopword_list)
    # print(len(stopword_list))
    word_cloud = WordCloud(
        background_color='white',
        stopwords=stopword_list,
        max_words=100,
        max_font_size=50,
        random_state=42
    ).generate(text)
    # print(word_cloud)
    fig = plt.figure(1)
    plt.imshow(word_cloud)
    plt.axis('off')
    # plt.show()
    fname = username + "_word.png"
    fig.savefig(fname, dpi=900)
    data = [[text]]
    df = pd.DataFrame(data, columns=['article_txt'])
    tf_idf = TfidfVectorizer(max_df=1, stop_words=stopword_list, max_features=10000, ngram_range=(1, 3))
    # Learn vocabulary and idf from training set.
    tf_idf.fit(df.article_txt)
    doc = pd.Series(text)
    # Transform documents to document-term matrix.
    doc_vector = tf_idf.transform(doc)
    sorted_items = sort_coo(doc_vector.tocoo())
    # extract only the top n; n here is 10
    feature_names = tf_idf.get_feature_names()
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
    return keywords
    # print("Keywords:")
    # for k in keywords:
    #     print(k, keywords[k])


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
