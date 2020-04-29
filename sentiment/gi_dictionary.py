import pandas as pd
import numpy as np
import os
from data_handler.models import Word
from sentiment.models import Category


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_dictionary():
    # read dictionary excel
    df = pd.read_excel(os.path.join(BASE_DIR, "inquireraugmented.xls"), skiprows=[1], usecols="A,C:GB")
    df = df.replace(np.nan, '', regex=True)
    for index, row in df.iterrows():
        word = str(row["Entry"]).strip()
        print("Adding word: " + word)
        word = word.split("#")[0].strip()
        word_obj = ""
        word_obj = Word.objects.filter(word=word).first()
        if not word_obj:
            word_obj = Word(word=word)
            word_obj.save()
        for col in df.columns:
            if not row[col]=='':
                print("Not nan!: " + str(row[col]))
                cat = ""
                cat = Category.objects.filter(name=col).first()
                if not cat:
                    cat = Category(name=col)
                    cat.save()
                word_obj.categories.add(cat)
                word_obj.save()
                print("Added category!")
        print("Added word!")


def get_sentiment_for(filename):
    # open file and create DF
    file = open(filename, "r")
    df = pd.read_csv(file, index_col=0)
    frequency_maps = df['frequency_map']
    distribution = get_distribution(frequency_maps)
