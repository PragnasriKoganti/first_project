import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('dataset/students.csv')

# Encode categorical features
le = LabelEncoder()
df['previous_grade'] = le.fit_transform(df['previous_grade'])
df['internet_access'] = le.fit_transform(df['internet_access'])
df['parent_education'] = le.fit_transform(df['parent_education'])

# Create target column (Pass/Fail)
df['performance'] = df['final_score'].apply(lambda x: 'Fail' if x < 40 else 'Pass')

# Features and target
X = df.drop(['final_score', 'performance'], axis=1)
y = df['performance']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, 'student_model.pkl')
