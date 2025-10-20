import pandas as pd
import joblib

def predict(file_path):
    # Load saved model
    model = joblib.load('models/student_performance_model.pkl')

    # Load data with correct separator
    df = pd.read_csv(file_path, sep=';')

    # Drop target if exists
    if 'G3' in df.columns:
        df = df.drop('G3', axis=1)

    # Predict
    preds = model.predict(df)
    print("🎯 Sample Predictions:", preds[:10])
    return preds

if __name__ == "__main__":
    predict('data/student-mat.csv')
