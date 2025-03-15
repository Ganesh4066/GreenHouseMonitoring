import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import requests
import tflite_runtime.interpreter as tflite
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

app = Flask(__name__)
app.config['DEBUG'] = True

# Set up logging to output debug info
logging.basicConfig(level=logging.DEBUG)

# -------------------------------------------------
# Define file paths for preprocessing objects and dataset
# -------------------------------------------------
scaler_path = "scaler.pkl"
label_encoder_path = "label_encoder.pkl"
data_path = "Modified_Crop_Data_Cleaned.csv"  # Ensure this file is in your container

# Define the features and target based on your dataset
features = ["temperature", "ph", "humidity", "soil_moisture", "sunlight_exposure", "soil_type"]
target = "label"

# -------------------------------------------------
# Load or Fit Preprocessing Objects
# -------------------------------------------------
try:
    if os.path.exists(scaler_path) and os.path.exists(label_encoder_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(label_encoder_path, "rb") as f:
            le = pickle.load(f)
        logging.debug("Loaded pre-saved scaler and label encoder.")
    else:
        logging.debug("Pre-saved scaler and label encoder not found. Fitting from dataset.")
        df = pd.read_csv(data_path, encoding="latin1")
        df.fillna(0, inplace=True)
        df_train = df[features + [target]].copy()
        df_train[target] = df_train[target].astype(str)
        le = LabelEncoder()
        df_train[target] = le.fit_transform(df_train[target])
        scaler = StandardScaler()
        X_train = df_train[features].astype(np.float32)
        scaler.fit(X_train)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        with open(label_encoder_path, "wb") as f:
            pickle.dump(le, f)
        logging.debug("Fitted and saved scaler and label encoder.")
except Exception as e:
    logging.error(f"Error loading/fitting preprocessing objects: {e}")
    raise

# Build a label mapping from integer to crop name
label_mapping = {i: label for i, label in enumerate(le.classes_)}

# -------------------------------------------------
# Load TFLite Model Using Absolute Path
# -------------------------------------------------
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tflite_model_path = os.path.join(current_dir, "greenhouse_model.tflite")
    interpreter = tflite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.debug("TFLite model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading TFLite model: {e}")
    raise

# -------------------------------------------------
# Firebase URL for Sensor Data
# -------------------------------------------------
FIREBASE_SENSOR_URL = "https://green-house-monitoring-2a06d-default-rtdb.firebaseio.com/Greenhouse/SensorData.json"

# -------------------------------------------------
# Helper function: Fetch sensor data and run inference
# -------------------------------------------------
def get_sensor_and_inference():
    try:
        response = requests.get(FIREBASE_SENSOR_URL)
    except Exception as e:
        raise Exception(f"Error fetching sensor data: {e}")
    
    if response.status_code != 200:
        raise Exception("Failed to fetch sensor data from Firebase")
    
    sensor_data = response.json()
    if not sensor_data:
        raise Exception("No sensor data found in Firebase")
    
    try:
        temperature = float(sensor_data.get("temperature", 25.0))
        ph = float(sensor_data.get("pH", 7.0))
        humidity = float(sensor_data.get("humidity", 50.0))
        soil_moisture = float(sensor_data.get("soilMoisture", 0))
        sunlight_exposure = float(sensor_data.get("lux", 100))
    except ValueError as e:
        raise Exception(f"Invalid sensor value: {e}")
    
    mapped_features = {
        "temperature": temperature,
        "ph": ph,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "sunlight_exposure": sunlight_exposure
    }
    
    input_array = np.array([[temperature, ph, humidity, soil_moisture, sunlight_exposure]], dtype=np.float32)
    input_scaled = scaler.transform(input_array)
    
    try:
        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
    except Exception as e:
        raise Exception(f"TFLite inference failed: {e}")
    
    predicted_class = int(np.argmax(output_data[0]))
    predicted_label = label_mapping.get(predicted_class, "Unknown")
    
    return sensor_data, mapped_features, output_data, predicted_class, predicted_label

# -------------------------------------------------
# Flask Endpoints
# -------------------------------------------------
@app.route("/")
def home():
    return "Crop Prediction API using Firebase data is running."

@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        sensor_data, mapped_features, output_data, predicted_class, predicted_label = get_sensor_and_inference()
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500
    
    message = (
        f"Predicted crop: '{predicted_label}'. Conditions: Temperature = {mapped_features['temperature']}°C, "
        f"pH = {mapped_features['ph']}, Humidity = {mapped_features['humidity']}%, "
        f"Soil Moisture = {mapped_features['soil_moisture']}, "
        f"Sunlight Exposure = {mapped_features['sunlight_exposure']} lux."
    )
    
    response_json = {
        "sensor_data": sensor_data,
        "mapped_features": mapped_features,
        "predicted_class": predicted_class,
        "predicted_crop": predicted_label,
        "raw_model_output": output_data.tolist(),
        "message": message
    }
    return jsonify(response_json), 200

@app.route("/dashboard", methods=["GET"])
def dashboard():
    try:
        sensor_data, mapped_features, output_data, predicted_class, predicted_label = get_sensor_and_inference()
    except Exception as e:
        logging.error(e)
        return f"Error: {e}", 500
    
    html = """
    <!doctype html>
    <html>
      <head>
        <title>Greenhouse Monitoring Dashboard</title>
      </head>
      <body>
        <h1>Greenhouse Monitoring Dashboard</h1>
        <h2>Sensor Data</h2>
        <ul>
          <li>Temperature: {{ temperature }} °C</li>
          <li>pH: {{ ph }}</li>
          <li>Humidity: {{ humidity }} %</li>
          <li>Soil Moisture: {{ soil_moisture }}</li>
          <li>Sunlight Exposure: {{ sunlight_exposure }} lux</li>
        </ul>
        <h2>TFLite Model Output</h2>
        <ul>
          <li>Raw Model Output: {{ output_data }}</li>
          <li>Predicted Class: {{ predicted_class }}</li>
          <li>Predicted Crop: {{ predicted_label }}</li>
        </ul>
      </body>
    </html>
    """
    return render_template_string(html,
                                  temperature=mapped_features['temperature'],
                                  ph=mapped_features['ph'],
                                  humidity=mapped_features['humidity'],
                                  soil_moisture=mapped_features['soil_moisture'],
                                  sunlight_exposure=mapped_features['sunlight_exposure'],
                                  output_data=output_data.tolist(),
                                  predicted_class=predicted_class,
                                  predicted_label=predicted_label)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
