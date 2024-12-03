import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.models import FertilizerInput

def ensure_model_exists(dataset_path='r.csv'):
    """
    Check if model files exist, if not, train and save the model
    """
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/rf_model.pkl'
    scaler_path = 'models/scaler.pkl'
    encoders_path = 'models/label_encoders.pkl'
    
    # If any of the model files are missing, retrain the model
    if not (os.path.exists(model_path) and 
            os.path.exists(scaler_path) and 
            os.path.exists(encoders_path)):
        print("Model files not found. Training new model...")
        train_and_save_model(dataset_path)
    
    return model_path, scaler_path, encoders_path

def train_and_save_model(dataset_path='r.csv'):
    """
    Train and save the machine learning model
    """
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please provide a valid dataset path.")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Separate features and target
    X = df.drop(columns=['Fertilizer Name'])
    y = df['Fertilizer Name']
    
    # Encode categorical features
    label_encoders = {}
    categorical_columns = ['Soil Type', 'Crop Type']
    for column in categorical_columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and preprocessing objects
    joblib.dump(model, 'models/rf_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    print("Model training and saving completed!")

class FertilizerPredictionApp:
    def __init__(self, 
                 dataset_path='data/fertilizer_dataset.csv',
                 model_path=None, 
                 scaler_path=None, 
                 encoders_path=None):
        
        # Ensure model exists
        if model_path is None:
            model_path, scaler_path, encoders_path = ensure_model_exists(dataset_path)
        
        # Load pre-trained model and preprocessing objects
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)

    def preprocess_input(self, input_data: FertilizerInput):
        # Convert input to DataFrame
        df = pd.DataFrame([{
            'Soil Type': input_data.Soil_Type,
            'Crop Type': input_data.Crop_Type,
            'Nitrogen': input_data.Nitrogen,
            'Phosphorus': input_data.Phosphorus,
            'Potassium': input_data.Potassium,
            'Temperature': input_data.Temperature,
            'Humidity': input_data.Humidity,
            'Moisture': input_data.Moisture
        }])
        
        # Encode categorical features
        df['Soil Type'] = self.label_encoders['Soil Type'].transform(df['Soil Type'])
        df['Crop Type'] = self.label_encoders['Crop Type'].transform(df['Crop Type'])
        
        # Standardize features
        features = df[['Soil Type', 'Crop Type', 'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Moisture']]
        scaled_features = self.scaler.transform(features)
        
        return scaled_features

    def predict(self, input_data: FertilizerInput):
        # Preprocess input
        preprocessed_input = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(preprocessed_input)
        return prediction[0]
