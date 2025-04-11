import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("Cleaned_Dataset.csv")

# Define the mappings
mappings = {
    'Sex': {'M': 0, 'Male': 0, 'F': 1, 'Female': 1},
    'ChestPainType': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3},
    'RestingECG': {'Normal': 0, 'ST': 1,'LVH':2},
    'ExerciseAngina': {'N': 0, 'Y': 1}
}

# Apply the mappings
for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)


numerical_cols = ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'ST_Slope',
                  'trestbps', 'thalach', 'ca', 'thal', 'Merged_Age']
# Round numerical columns and convert to int
for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].round().astype(int)


# split the data into X and y
X= df.drop('HeartDisease', axis=1)
y = df['HeartDisease']


# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, 'models/heart_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Model & scaler saved.")
