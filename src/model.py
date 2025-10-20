import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def build_pipeline(model=None):
    if model is None:
        model = RandomForestRegressor(random_state=42)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def evaluate_model(pipeline, X, y, cv=5):
    scores = cross_val_score(pipeline, X, y, scoring='neg_root_mean_squared_error', cv=cv)
    return -scores.mean()

def save_model(pipeline, path='models/final_model.joblib'):
    joblib.dump(pipeline, path)
