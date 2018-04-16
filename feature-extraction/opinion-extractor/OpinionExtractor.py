import json
import os
import numpy as np
import pandas as pd
import nltk


class OpinionExtractor:

    def initialize_subjectivity():

        sj = []
        with open('./subjectivity.tff', 'r') as subject:
            sj = subject.readlines()

        subj = []
        for element in sj:
            subj.append(element.replace('\n', ''))

        word_list = []
        wtype_list = []
        pos_list = []
        polar_list = []

        for txt in subj:
            #word
            word = txt.split(' ')[2].split('=')[1]

            #type
            wtype = txt.split(' ')[0].split('=')[1]

            #pos
            pos = txt.split(' ')[3].split('=')[1]

            #polarity
            polar = txt.split(' ')[5].split('=')[1]

            word_list.append(word)
            wtype_list.append(wtype)
            pos_list.append(pos)
            polar_list.append(polar)  
            
        dat = {'word' : word_list, 'type':wtype_list, 'pos':pos_list, 'polarity':polar_list}
        subjectivity = pd.DataFrame(data = dat)

        strongly_subj_list = subjectivity[subjectivity['type'] == 'strongsubj']['word'].tolist()

        return strongly_subj_list

    def opinion_get(row):
        
        global strongly_subj_list
        
        text = row['text']
        text_words = nltk.word_tokenize(text.lower())
        text_words = [word for word in text_words if word.isalpha()]

        opinion = 0
        
        for word in text_words:
            if word in strongly_subj_list:
                opinion = 1
        
        return opinion


    def add_opinion_column(df):
        df['opinion'] = df.apply(lambda x: FileReader.opinion_get(x), axis = 1)




