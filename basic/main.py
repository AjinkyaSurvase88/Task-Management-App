import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

# Initialise empty task list
tasks = pd.DataFrame(columns=['description','priority'])

# load pre-existing tasks
try:
    tasks=pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

#to save tasks to csv file
def save_tasks():
    tasks.to_csv('tasks.csv', index=False)
    
# train the task priority classifier
vectorizer = CountVectorizer()
clf= MultinomialNB()
model=make_pipeline(vectorizer,clf)
model.fit(tasks['description'],tasks['priority'])
 
