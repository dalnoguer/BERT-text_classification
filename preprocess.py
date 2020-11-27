# import re

# reviews_train = []
# for line in open('/home/dal/datasets/imdb/movie_data/full_train.txt', 'r'):
    
#     reviews_train.append(line.strip())
    
# reviews_test = []
# for line in open('/home/dal/datasets/imdb/movie_data/full_test.txt', 'r'):
    
#     reviews_test.append(line.strip())


# import re
# REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
# NO_SPACE = ""
# SPACE = " "

# def preprocess_reviews(reviews):
    
#     #reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
#     reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
#     return reviews

# reviews_train_clean = preprocess_reviews(reviews_train)
# reviews_test_clean = preprocess_reviews(reviews_test)

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# df_train_reviews_clean = pd.DataFrame(reviews_train_clean, columns =['reviews'])
# df_train_reviews_clean['target'] = np.where(df_train_reviews_clean.index<12500,1,0)
# df_test_reviews_clean = pd.DataFrame(reviews_test_clean, columns =['reviews'])
# df_test_reviews_clean['target'] = np.where(df_test_reviews_clean.index<12500,1,0)
# # Shuffling the rows in both the train and test data.  This is very important before using the data for training.
# df_train_reviews_clean = df_train_reviews_clean.sample(frac=1).reset_index(drop=True)
# df_test_reviews_clean = df_test_reviews_clean.sample(frac=1).reset_index(drop=True)
# # breaking the train data into training and validation
# df_train, df_valid = train_test_split(df_train_reviews_clean, test_size=0.25, stratify=df_train_reviews_clean['target'])
# df_train.reset_index(drop=True, inplace=True)
# df_valid.reset_index(drop=True, inplace=True)
# df_test = df_test_reviews_clean

# df_train.to_csv('/home/dal/datasets/imdb/preprocessed/train.csv', index=False)
# df_valid.to_csv('/home/dal/datasets/imdb/preprocessed/valid.csv', index=False)
# df_test.to_csv('/home/dal/datasets/imdb/preprocessed/test.csv', index=False)

import pandas as pd
df_train = pd.read_csv('/home/dal/datasets/imdb/preprocessed/train.csv')
df_valid = pd.read_csv('/home/dal/datasets/imdb/preprocessed/valid.csv')
df_test = pd.read_csv('/home/dal/datasets/imdb/preprocessed/test.csv')

breakpoint()