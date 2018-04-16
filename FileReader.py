import json
import os
import numpy as np
import pandas as pd
import nltk

train_path = 'data/semeval2017-task8-dataset/traindev/rumoureval-subtaskB-train.json'
dev_path = 'data/semeval2017-task8-dataset/traindev/rumoureval-subtaskB-dev.json' 
eval_path = 'data/semeval2017-task8-dataset/rumoureval-data/'

test_folder_path = 'data/semeval2017-task8-test-data/'

strongly_subj_list = None

class FileReader:
    def exec(self):
        train_data_simple, dev_data_simple, train_df_simple, dev_df_simple = FileReader.load_train_dev(train_path, dev_path, eval_path, simple=True)
        test_data_simple, test_df_simple = FileReader.load_test_data(test_folder_path, simple=True)

        train_data_full, dev_data_full, train_df_full, dev_df_full = FileReader.load_train_dev(train_path, dev_path, eval_path, simple=False)
        test_data_full, test_df_full = FileReader.load_test_data(test_folder_path, simple=False)
        
        data_list = [train_data_simple, dev_data_simple, test_data_simple, train_data_full, dev_data_full, test_data_full]
        df_list = [train_df_simple, dev_df_simple, test_df_simple, train_df_full, dev_df_full, test_df_full]

        data_name_list = ['train_data_simple', 'dev_data_simple', 'test_data_simple', 'train_data_full', 'dev_data_full', 'test_data_full']

        print('saving data to output..')
        
        # create output folder
        output_folder_path = './output'
        simple_path = output_folder_path + '/simple'
        full_path = output_folder_path + '/full'
        folder_list = [output_folder_path, simple_path, full_path]
        for folder in folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # save data
        for idx in range(len(data_list)):
            
            data_name = data_name_list[idx]
            if '_simple' in data_name:        
                json_output_name = simple_path + '/' + data_name_list[idx] + '.json'
                pickle_output_name = simple_path + '/' + data_name_list[idx] + '.pickle'
            else:        
                json_output_name = full_path + '/' + data_name_list[idx] + '.json'
                pickle_output_name = full_path + '/' + data_name_list[idx] + '.pickle'
                
            with open(json_output_name, 'w') as f:
                f.write(json.dumps(data_list[idx]))
                
            df_list[idx].to_pickle(pickle_output_name)

        # TEST CODE FOR READING PANDAS DATAFRAME
        print('test code: sample of dev data (simple version) \n')
        df = pd.read_pickle('./output/simple/dev_data_simple.pickle')

        #preprocess the text column so that @xxx -> @someuser and http:// -> someurl
        # preprocess(df)

        #initialize list of strongly subjective words
        global strongly_subj_list
        strongly_subj_list = FileReader.initialize_subjectivity()

        #add a binary column where opinion == 1 if the tweet text contains a strongly subjective word
        FileReader.add_opinion_column(df)

        print(df[:2])
        return df        

    #def __init__(self):


    def replace_url_at(row):
        text = row['text']
        new_list = []
        word_list = text.split()
        for word in word_list:
            if word[0] == '@':
                new_list.append('@someuser')
            elif word[:7] == 'http://':
                new_list.append('someurl')
            else:
                new_list.append(word)
                
        new_str = ' '.join(new_list)
        new_str = new_str.replace('\n', ' ').replace('\t', ' ')
        return new_str

    def preprocess(df):
        df['text'] = df.apply(FileReader.replace_url_at, axis = 1)


    def source_tweet_data(tweet_id, folder_path_dict, simple=False):
        tweet_data = {}
        
        folder_path = folder_path_dict[tweet_id]
        source_tweet_path = folder_path + 'source-tweet/' + tweet_id + '.json'
        
        with open(source_tweet_path, 'r') as f:
            source_tweet_str = f.read()

        source_tweet = json.loads(source_tweet_str)    
            
        # Take only the text data
        # FOR BASELINE IMPLEMENTATION
        if simple:
            tweet_data['text'] = source_tweet['text']
        
        # Use all information in the source tweet
        # NOT USED FOR BASELINE, BUT MAY BE USEFUL FOR GENERATING MORE FEATURES
        else:
            tweet_data = source_tweet
        
        # TODO: EXTRACT INFORMATION FROM TWEET REPLIES
        
        # does the tweet have context? (boolean value)
        has_context = int(os.path.isdir(folder_path + 'context'))
        tweet_data['has_context'] = has_context
        
        # if it does, point to context path
        if has_context:
            tweet_data['context_path'] = folder_path + 'context/'
        else:
            tweet_data['context_path'] = np.nan    
        
        return tweet_data

    # utility to remove .DS_Store files
    def pruneOSXArtifactFiles(a_list):
        if '.DS_Store' in a_list:
            a_list.remove('.DS_Store')
        return a_list


    def load_train_dev(train_path, dev_path, eval_path, simple=True):

        with open(train_path, 'r') as f1, open(dev_path, 'r') as f2:
            train_str = f1.read()
            dev_str = f2.read()

        train_dict = json.loads(train_str)    

        dev_dict = json.loads(dev_str)
        train_data = {}
        dev_data = {}

        folder_path_dict = {}
        
        topic_list = os.listdir(eval_path)
        topic_dict = {}

        # maintain folder path dictionary to use during feature generation
        for topic in topic_list:

            for tweet_id in os.listdir(eval_path + topic):

                # keep track of topic-id pairs
                topic_dict[tweet_id] = topic

                # add id-folderpath pairs
                folder_path_dict[tweet_id] = eval_path + topic + '/' + tweet_id + '/'

        # generate features for training data
        for tweet_id in train_dict.keys():
            train_data[tweet_id] = FileReader.source_tweet_data(tweet_id, folder_path_dict, simple)
            
            # note that the test data does not have explicit topic labels and
            # therefore must have a separate process to extract topic from them
            train_data[tweet_id]['topic'] = topic_dict[tweet_id]
            train_data[tweet_id]['classification'] = train_dict[tweet_id]

        # generate features for dev data
        for tweet_id in dev_dict.keys():
            dev_data[tweet_id] = FileReader.source_tweet_data(tweet_id, folder_path_dict, simple)
            dev_data[tweet_id]['topic'] = dev_dict[tweet_id]
            dev_data[tweet_id]['classification'] = dev_dict[tweet_id]        
        
        # save as pandas dataframe
        train_df = pd.DataFrame(train_data).transpose()
        dev_df = pd.DataFrame(dev_data).transpose()
        
        return train_data, dev_data, train_df, dev_df


    def load_test_data(test_folder_path, simple=True):
        
        test_data = {}    
        folder_path_dict = {}

        # maintain folder path dictionary to use during feature generation
        for tweet_id in os.listdir(test_folder_path):
            # add id-folderpath pairs
            folder_path_dict[tweet_id] = test_folder_path + '/' + tweet_id + '/'
        
        # generate features for test data
        for tweet_id in FileReader.pruneOSXArtifactFiles(os.listdir(test_folder_path)):
            print("\n",tweet_id)
            test_data[tweet_id] = FileReader.source_tweet_data(tweet_id, folder_path_dict, simple)
        
        # save as pandas dataframe
        test_df = pd.DataFrame(test_data).transpose()
            
        return test_data, test_df

    def initialize_subjectivity():

        sj = []
        with open('./feature-extraction/opinion-extractor/subjectivity.tff', 'r') as subject:
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



