import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Synthetic dataset
data = {
    'age': [30, 45, 28, 60, 35, 50, 40, 42, 29, 55],
    'job': ['admin.', 'technician', 'blue-collar', 'retired', 'admin.', 'services', 'management', 'unemployed', 'technician', 'retired'],
    'marital': ['married', 'single', 'single', 'married', 'divorced', 'married', 'single', 'divorced', 'single', 'married'],
    'education': ['tertiary', 'secondary', 'secondary', 'primary', 'tertiary', 'secondary', 'tertiary', 'primary', 'tertiary', 'primary'],
    'balance': [1000, 200, -50, 1500, 800, 300, 700, 0, -200, 100],
    'housing': ['yes', 'no', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'no'],
    'loan': ['no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no'],
    'contact': ['cellular', 'telephone', 'cellular', 'cellular', 'telephone', 'telephone', 'cellular', 'cellular', 'telephone', 'cellular'],
    'duration': [300, 120, 250, 400, 150, 180, 210, 220, 100, 350],
    'campaign': [2, 3, 1, 2, 1, 2, 1, 3, 1, 1],
    'y': ['yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'yes']
}

df = pd.DataFrame(data)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split data
X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build and train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
