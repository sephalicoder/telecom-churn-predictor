import joblib
from xgboost import XGBClassifier

# Load your trained model
print("Loading local model...")
xgb = joblib.load('xgb_churn_model.pkl')

# Save a clean version inside the directory for the app to read
joblib.dump(xgb, 'model_deploy.pkl')
print("Saved model_deploy.pkl successfully! Ready for Streamlit. ✅")
