import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import os

# Create folder if not exists
os.makedirs('models', exist_ok=True)

# 1️⃣ Load dataset
df = pd.read_csv('data/student-mat.csv', sep=';')

# 2️⃣ Split into features (X) and target (y)
X = df.drop('G3', axis=1)
y = df['G3']

# 3️⃣ Identify column types
cat = X.select_dtypes(include=['object']).columns
num = X.select_dtypes(exclude=['object']).columns

# 4️⃣ Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
])

# 5️⃣ Define model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 6️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ Train the model
model.fit(X_train, y_train)

# 8️⃣ Predictions
y_pred = model.predict(X_test)

# 9️⃣ Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Model Training Completed!")
print("📊 Evaluation Results:")
print(f"   MAE  (Mean Absolute Error): {mae:.2f}")
print(f"   RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"   R²   (Coefficient of Determination): {r2:.2f}")

# 🔟 Save trained model
joblib.dump(model, 'models/student_performance_model.pkl')
print("\n💾 Model saved to 'models/student_performance_model.pkl'")
