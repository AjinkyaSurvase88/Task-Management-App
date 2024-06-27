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
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)
model.fit(tasks['description'], tasks['priority'])

 
# fuction adding task to list
def add_task(description,priority):
    global tasks
    new_task=pd.DataFrame({'description':[description],'priority':priority})
    tasks=pd.concat([tasks,new_task],ignore_index=True)
    save_tasks()
    
# function to remove task
def remove_task(description):
    global tasks
    tasks=tasks[tasks['description'] !=description]
    save_tasks()

# list the tasks
def list_tasks():
    if tasks.empty:
        print("no task available")
    else:
        print(tasks)
    
# Function to recommend task based on ML
def recommend_task():
    if not tasks.empty:
        #get high-priority tasks
        high_priority_tasks = tasks[tasks['priority']=='High']
        
        if not high_priority_tasks.empty:
            # choose random high priority task
            random_task = random.choice(high_priority_tasks['description'].tolist())
            print(f"Recommend task: {random_task} - priority :'high")
            
        else:
            print("no high priority task available")
            
    else:
        print("no task available for recommendation")
        
# Main menu 
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Task")
    print("4. Recommend Task")
    print("5. Exit")
    
    choice=input("select an option: ")
    
    if choice=="1":
        description=input("Enter a task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        add_task(description,priority)
        print("Task added succesfully")
        
    elif choice =="2":
        description=input("Enter a task to remove: ")
        remove_task(description)
        print("Task removed succesfully")
        
    elif choice =="3":
        list_tasks()
    
    elif choice =="4":
        recommend_task()
        
    elif choice =="5":
        print("Good bye!")
        break
    else:
        print("Invalid Option. Please select valid option. ")

