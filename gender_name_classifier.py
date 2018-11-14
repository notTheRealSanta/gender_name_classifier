import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Importing data
male_names_set = pd.read_csv('dataset/Indian-Male-Names.csv')
female_names_set = pd.read_csv('dataset/Indian-Female-Names.csv')

del male_names_set['race']
del female_names_set['race']

# removing second name from data set
male_names_set['first_name'] = male_names_set['name'].astype(str).str.split().str[0]
female_names_set['first_name'] = female_names_set['name'].astype(str).str.split().str[0]

# gender_  as (0/1) ( 1 -> Female )
male_names_set['gender_'] = 0
female_names_set['gender_'] = 1

del male_names_set['name']
del male_names_set['gender']
del female_names_set['name']
del female_names_set['gender']

# dataset -> first_name (str) , gender_ (0/1) ( 1 -> Female )
data_set = pd.concat([male_names_set, female_names_set])
data_set = shuffle(data_set)

# Data cleaning
data_set = data_set[data_set['first_name'].map(len) > 4]
data_set = data_set[data_set['first_name'].str.isalpha()]

# feature extraction
def input_function(names):
    train_input = np.zeros((len(names), 4))
    for idx, name in enumerate(names):
        try:
            # last character
            train_input[idx, 0] = ord(name[-1]) - ord('a')
            # second last character
            train_input[idx, 1] = ord(name[-2]) - ord('a')
            # first character
            train_input[idx, 2] = ord(name[0]) - ord('a')
            # second character
            train_input[idx, 3] = ord(name[1]) - ord('a')
        except:
            print(name)

    return train_input


# Separating input and output
train_input = input_function(list(data_set['first_name'].tolist()))
train_output = data_set['gender_'].values

# Classifier
clf = RandomForestClassifier(n_estimators=200, min_samples_split=10)

# Training
print("training...")
clf.fit(train_input, train_output)

print("^C to exit\n")

while (1):
    input_string = input("Name : ")
    result = (clf.predict(input_function([input_string.lower()])))
    if result == 1:
        print("Female")
    else:
        print("Male")
