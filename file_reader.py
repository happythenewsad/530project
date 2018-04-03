
# coding: utf-8

# In[78]:


import json
import os
import numpy as np
import pandas as pd

train_path = './semeval2017-task8-dataset/traindev/rumoureval-subtaskB-train.json'
dev_path = './semeval2017-task8-dataset/traindev/rumoureval-subtaskB-dev.json' 
eval_path = './semeval2017-task8-dataset/rumoureval-data/'

test_folder_path = './semeval2017-task8-test-data/'

def source_tweet_data(tweet_id, folder_path_dict, simple=True):
    tweet_data = {}
    
    folder_path = folder_path_dict[tweet_id]
    source_tweet_path = folder_path + 'source-tweet/' + tweet_id + '.json'
    
    with open(source_tweet_path, 'r') as f:
        source_tweet_str = f.read()

    source_tweet = json.loads(source_tweet_str)    
        
    #Take only the text data
    #FOR BASELINE IMPLEMENTATION
    if simple:
        tweet_data['text'] = source_tweet['text']
    
    #Use all information in the source tweet 
    #NOT USED FOR BASELINE, BUT MAY BE USEFUL FOR GENERATING MORE FEATURES
    else:
        tweet_data = source_tweet
    
    #TODO: EXTRACT INFORMATION FROM TWEET REPLIES
    
    #does the tweet have context? (boolean value)
    has_context = os.path.isdir(folder_path + 'context')    
    tweet_data['has_context'] = has_context
    
    #if it does, point to context path
    if has_context:
        tweet_data['context_path'] = folder_path + 'context/'
    else:
        tweet_data['context_path'] = np.nan    
    
    return tweet_data

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

    #maintain folder path dictionary to use during feature generation
    for topic in topic_list:

        for tweet_id in os.listdir(eval_path + topic):

            #keep track of topic-id pairs 
            topic_dict[tweet_id] = topic

            #add id-folderpath pairs
            folder_path_dict[tweet_id] = eval_path + topic + '/' + tweet_id + '/'
            
      
    #generate features for training data
    for tweet_id in train_dict.keys():
        train_data[tweet_id] = source_tweet_data(tweet_id, folder_path_dict, simple)
        
        #note that the test data does not have explicit topic labels and therefore must have a separate process to extract topic from them
        train_data[tweet_id]['topic'] = topic_dict[tweet_id]
        train_data[tweet_id]['classification'] = train_dict[tweet_id]

    #generate features for dev data
    for tweet_id in dev_dict.keys():
        dev_data[tweet_id] = source_tweet_data(tweet_id, folder_path_dict, simple)
        dev_data[tweet_id]['topic'] = dev_dict[tweet_id]
        dev_data[tweet_id]['classification'] = dev_dict[tweet_id]        
    
    #save as pandas dataframe
    train_df = pd.DataFrame(train_data).transpose()
    dev_df = pd.DataFrame(dev_data).transpose()
    
    return train_data, dev_data, train_df, dev_df

def load_test_data(test_folder_path, simple=True):
    
    test_data = {}    
    folder_path_dict = {}

    #maintain folder path dictionary to use during feature generation
    for tweet_id in os.listdir(test_folder_path):
        #add id-folderpath pairs
        folder_path_dict[tweet_id] = test_folder_path + '/' + tweet_id + '/'
    
    #generate features for test data
    for tweet_id in os.listdir(test_folder_path):
        test_data[tweet_id] = source_tweet_data(tweet_id, folder_path_dict, simple)
        
    
    #save as pandas dataframe
    test_df = pd.DataFrame(test_data).transpose()
        
    return test_data, test_df

if __name__ == "__main__":

    print('generating features..')
    train_data_simple, dev_data_simple, train_df_simple, dev_df_simple = load_train_dev(train_path, dev_path, eval_path, simple=True)
    test_data_simple, test_df_simple = load_test_data(test_folder_path, simple=True)

    train_data_full, dev_data_full, train_df_full, dev_df_full = load_train_dev(train_path, dev_path, eval_path, simple=False)
    test_data_full, test_df_full = load_test_data(test_folder_path, simple=False)
    
    data_list = [train_data_simple, dev_data_simple, test_data_simple, train_data_full, dev_data_full, test_data_full]
    df_list = [train_df_simple, dev_df_simple, test_df_simple, train_df_full, dev_df_full, test_df_full]

    data_name_list = ['train_data_simple', 'dev_data_simple', 'test_data_simple', 'train_data_full', 'dev_data_full', 'test_data_full']

    
    print('saving data to output..')
    
    #crete output folder
    output_folder_path = './output'
    simple_path = output_folder_path + '/simple'
    full_path = output_folder_path + '/full'
    folder_list = [output_folder_path, simple_path, full_path]
    for folder in folder_list:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    #save data
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
        

    print('test code: sample of dev data (full version) \n')    
    #TEST CODE FOR READING PANDAS DATAFRAME
    df = pd.read_pickle('./output/full/dev_data_full.pickle')
    print(df.head())
    print('\n\n')
    
    #TEST CODE FOR READING JSON AS DICT
    with open('./output/full/dev_data_full.json', 'r') as f:
        jstr = f.read()

    j = json.loads(jstr)
    
    print(next(iter(j.values())))

