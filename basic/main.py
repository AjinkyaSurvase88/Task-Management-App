import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random
import streamlit as st

# Initialize empty task list
tasks = pd.DataFrame(columns=['description', 'priority'])

# Load pre-existing tasks
try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

# Function to save tasks to CSV file
def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

# Train the task priority classifier
def train_model():
    if not tasks.empty:
        vectorizer = CountVectorizer()
        clf = MultinomialNB()
        model = make_pipeline(vectorizer, clf)
        model.fit(tasks['description'], tasks['priority'])
        return model
    else:
        return None

model = train_model()

# Function to add task to list
def add_task(description, priority):
    global tasks
    new_task = pd.DataFrame({'description': [description], 'priority': priority})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()

# Function to remove task
def remove_task(description):
    global tasks
    tasks = tasks[tasks['description'] != description]
    save_tasks()

# Function to list the tasks
def list_tasks():
    if tasks.empty:
        st.write("No tasks available")
    else:
        st.write(tasks)

# Function to recommend task based on ML
def recommend_task():
    if not tasks.empty:
        high_priority_tasks = tasks[tasks['priority'] == 'High']
        if not high_priority_tasks.empty:
            random_task = random.choice(high_priority_tasks['description'].tolist())
            st.write(f"Recommended task: {random_task} - Priority: High")
        else:
            st.write("No high priority tasks available")
    else:
        st.write("No tasks available for recommendation")

# Streamlit UI
st.title("Task Management App")

menu = ["Add Task", "Remove Task", "List Tasks", "Recommend Task"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Add Task":
    st.subheader("Add Task")
    description = st.text_input("Enter task description")
    priority = st.selectbox("Enter task priority", ["Low", "Medium", "High"])
    if st.button("Add Task"):
        add_task(description, priority)
        st.success("Task added successfully")

elif choice == "Remove Task":
    st.subheader("Remove Task")
    description = st.text_input("Enter task description to remove")
    if st.button("Remove Task"):
        remove_task(description)
        st.success("Task removed successfully")

elif choice == "List Tasks":
    st.subheader("Task List")
    list_tasks()

elif choice == "Recommend Task":
    st.subheader("Recommended Task")
    recommend_task()
