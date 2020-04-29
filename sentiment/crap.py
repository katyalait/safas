# first i loaded the model
from sentiment.model import time_to_complete
import multiprocessing as mp
from gensim.models import Word2Vec
import os
# BASE = os.path.dirname(os.path.abspath(__file__))
# CLEANED_DATA = os.path.join(BASE, "cleaned_data.csv")
# MODEL = os.path.join(BASE, "model2vec.model")
from data_handler.models import Category

model = Word2Vec.load("word2vec.model")
# then i got the word vector from it
wv = model.wv
# then i got the negative + positive words in our store of negative words
neg = Category.objects.get(name="Negativ")
neg_words = neg.words.all()

pos = Category.objects.get(name="Positiv")
pos_words = pos.words.all()

neg_list = [word.word.lower() for word in neg_words]
pos_list = [word.word.lower() for word in pos_words]


subjective_words = neg_list + pos_list

# then i got all these words and constructed a dictionary from them and
# their associated words that were found in the corpus
dictionary = {}
dictionary = set()
for word in subjective_words:
    if word in wv:
        [dictionary.add(x) for (x,y) in wv.most_similar(positive=[word])]
        dictionary.add(word)

word_list = list(dictionary)
# then i constructed an association matrix



def similarity_matrix(wv, word_list):

    for word in word_list:
        vector = [wv.similarity(word, other_word) for other_word in word_list]
        matrix[word] = vector
    return matrix
index = 0
length = len(word_list)
time_ = time()
for word in word_list:
    index += 1
    time_remaining = round((0.5/60)*length, 2)
    vector = [wv.similarity(word, other_word) for other_word in word_list]
    if index%10==0:
        time_remaining = time_to_complete(index, time_, 10, length)
        time_ = time()
    progress(index, length, status="Time to complete {} mins".format(time_remaining))
    matrix[word] = vector
