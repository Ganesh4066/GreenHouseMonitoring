import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import requests
import tflite_runtime.interpreter as tflite
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Try to load pre-saved scaler and label encoder
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
except FileNotFoundError:
    # If not found, exit or handle appropriately
    print("Pre-saved scaler and label encoder not found.")
    exit(1)

label_mapping = {i: label for i, label in enumerate(le.classes_)}

# If you still need to load the dataset for any reason, check if it exists:
data_path = "Modified_Crop_Data_Cleaned.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path, encoding="latin1")
    df.fillna(0, inplace=True)
    features = ["temperature", "ph", "humidity", "soil_moisture", "sunlight_exposure", "soil_type"]
    target = "label"
    # You might use this to update ideal conditions, etc.
else:
    print("Dataset file not found; skipping dataset-based preprocessing.")

# Load TFLite model
current_dir = os.path.dirname(os.path.abspath(__file__))
tflite_model_path = os.path.join(current_dir, "greenhouse_model.tflite")
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Firebase URL (unchanged)
FIREBASE_SENSOR_URL = "https://green-house-monitoring-2a06d-default-rtdb.firebaseio.com/Greenhouse/SensorData.json"

@app.route("/")
def home():
    return "Crop Prediction API using Firebase data is running."

@app.route("/predict", methods=["POST"])
def predict():
    response = requests.get(FIREBASE_SENSOR_URL)
    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch sensor data from Firebase"}), 500
    sensor_data = response.json()
    if not sensor_data:
        return jsonify({"error": "No sensor data found in Firebase"}), 400

    try:
        temperature = float(sensor_data.get("temperature", 25.0))
        ph = float(sensor_data.get("pH", 7.0))
        humidity = float(sensor_data.get("humidity", 50.0))
        soil_moisture = float(sensor_data.get("soilMoisture", 0))
        sunlight_exposure = float(sensor_data.get("lux", 100))
    except ValueError as e:
        return jsonify({"error": f"Invalid sensor value: {e}"}), 400

    mapped_features = {
        "temperature": temperature,
        "ph": ph,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "sunlight_exposure": sunlight_exposure
    }

    input_array = np.array([[
        temperature, ph, humidity, soil_moisture, sunlight_exposure
    ]], dtype=np.float32)
    input_scaled = scaler.transform(input_array)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = int(np.argmax(output_data[0]))
    predicted_label = label_mapping.get(predicted_class, "Unknown")

    message = (
        f"Predicted crop: '{predicted_label}'. Conditions: Temperature = {temperature}°C, "
        f"pH = {ph}, Humidity = {humidity}%, Soil Moisture = {soil_moisture}, "
        f"Sunlight Exposure = {sunlight_exposure} lux."
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

if __name__ == "__main__":
    app.run(debug=False, port=5000)
