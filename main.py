import eel
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Initialize Eel with the 'web' folder
eel.init('web')

# Load and prepare the dataset
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df['Skills'] = df['Skills'].apply(lambda x: x.split(','))
    return df

df = load_and_prepare_data('salary_data.csv')

@eel.expose
def get_unique_values():
    unique_genders = df['Gender'].unique().tolist()
    unique_education_levels = df['Education Level'].unique().tolist()
    unique_job_titles = df['Job Titles'].unique().tolist()

    return {
        'genders': unique_genders,
        'education_levels': unique_education_levels,
        'job_titles': unique_job_titles
    }

# Train and save the model
def train_and_save_model(df):
    X = df[['Age', 'Gender', 'Education Level', 'Job Titles', 'Years of Experience']]
    y = df['Salary']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age', 'Years of Experience']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Education Level', 'Job Titles'])
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor())
    ])

    model_pipeline.fit(X_train, y_train)
    joblib.dump(model_pipeline, 'model_pipeline.pkl')

# Check if the model file exists, if not, train and save the model
if not os.path.exists('model_pipeline.pkl'):
    print("Model file not found. Training and saving the model...")
    train_and_save_model(df)

model_pipeline = joblib.load('model_pipeline.pkl')

# Setup SQLite database
def setup_database():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone_number TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

@eel.expose
def generate_learning_path(job_title, skills_textfield):
    def get_learning_paths_from_data(job_title):
        learning_paths = {}
        job_data = df[df['Job Titles'] == job_title]
        
        if not job_data.empty:
            levels = ['Beginner', 'Intermediate', 'Advanced']
            
            for level in levels:
                level_skills = set()  # Use a set to store unique skills
                for _, row in job_data.iterrows():
                    skills = [skill.strip() for skill in row['Skills']]
                    level_skills.update(skills)
                learning_paths[level] = list(level_skills)
        
        return learning_paths

    def assess_user_skills(skills_list, job_title):
        path = get_learning_paths_from_data(job_title)
        user_level = 'Beginner'
        levels = ['Advanced', 'Intermediate', 'Beginner']
        for level in levels:
            if all(skill in skills_list for skill in path.get(level, [])):
                user_level = level
                break
        return user_level

    def generate_learning_path(user_level, job_title, skills_list):
        path = get_learning_paths_from_data(job_title)
        levels = ['Beginner', 'Intermediate', 'Advanced']
        start_index = levels.index(user_level) + 1
        recommended_path = set()  # Use a set to ensure unique skills

        for level in levels[start_index:]:
            level_skills = path.get(level, [])
            # Ensure to exclude all skills the user already possesses
            filtered_skills = [skill for skill in level_skills if skill.strip() not in skills_list]
            recommended_path.update(filtered_skills)  # Update the set with new skills
        
        return list(recommended_path)

    # Process skills_textfield to handle both list and string types
    if isinstance(skills_textfield, str):
        skills_list = [skill.strip() for skill in skills_textfield.split(',')]
    elif isinstance(skills_textfield, list):
        skills_list = [skill.strip() for skill in skills_textfield]
    else:
        skills_list = []  # Default to an empty list if the type is unexpected

    user_level = assess_user_skills(skills_list, job_title)
    learning_path = generate_learning_path(user_level, job_title, skills_list)
    return learning_path







# Create a new user
@eel.expose
def create_user(first_name, last_name, email, phone_number, username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users (first_name, last_name, email, phone_number, username, password) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (first_name, last_name, email, phone_number, username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

# Authenticate the user
@eel.expose
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

# Predict salary based on user input
@eel.expose
def predict_salary(age, gender, education_level, job_title, experience):
    input_data = pd.DataFrame([[age, gender, education_level, job_title, experience]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Titles', 'Years of Experience'])
    predicted_salary = model_pipeline.predict(input_data)[0]
    return round(predicted_salary, 2)

# Start the app
if __name__ == '__main__':
    setup_database()
    eel.start('index.html')
