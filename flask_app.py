import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import requests
import tflite_runtime.interpreter as tflite
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# -------------------------------------------------
# Define file paths for preprocessing objects and dataset
# -------------------------------------------------
scaler_path = "scaler.pkl"
label_encoder_path = "label_encoder.pkl"
data_path = "Modified_Crop_Data_Cleaned.csv"  # CSV file must be included in your repo

# Define the features and target based on your dataset
# Note: If your TFLite model was trained on only 5 features, consider removing "soil_type"
features = ["temperature", "ph", "humidity", "soil_moisture", "sunlight_exposure", "soil_type"]
target = "label"

# -------------------------------------------------
# Load or Fit Preprocessing Objects
# -------------------------------------------------
if os.path.exists(scaler_path) and os.path.exists(label_encoder_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)
    print("Loaded pre-saved scaler and label encoder.")
else:
    print("Pre-saved scaler and label encoder not found. Fitting from dataset.")
    # Load the dataset and fit the scaler and label encoder
    df = pd.read_csv(data_path, encoding="latin1")
    df.fillna(0, inplace=True)
    df_train = df[features + [target]].copy()
    df_train[target] = df_train[target].astype(str)
    le = LabelEncoder()
    df_train[target] = le.fit_transform(df_train[target])
    scaler = StandardScaler()
    X_train = df_train[features].astype(np.float32)
    scaler.fit(X_train)
    # Save the fitted objects for future use
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(label_encoder_path, "wb") as f:
        pickle.dump(le, f)
    print("Fitted and saved scaler and label encoder.")

# Build a label mapping from integer to crop name
label_mapping = {i: label for i, label in enumerate(le.classes_)}

# -------------------------------------------------
# Load TFLite Model Using Absolute Path
# -------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
tflite_model_path = os.path.join(current_dir, "greenhouse_model.tflite")
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------------------------
# Firebase URL for Sensor Data
# -------------------------------------------------
FIREBASE_SENSOR_URL = "https://green-house-monitoring-2a06d-default-rtdb.firebaseio.com/Greenhouse/SensorData.json"

# -------------------------------------------------
# Flask Endpoints
# -------------------------------------------------
@app.route("/")
def home():
    return "Crop Prediction API using Firebase data is running."

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Fetches sensor data from Firebase, maps and scales it,
    runs inference with the TFLite model, and returns a JSON response.
    """
    # Fetch sensor data from Firebase
    try:
        response = requests.get(FIREBASE_SENSOR_URL)
    except Exception as e:
        return jsonify({"error": f"Error fetching sensor data: {e}"}), 500

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

    # Create input array in the expected order for the model
    input_array = np.array([[temperature, ph, humidity, soil_moisture, sunlight_exposure]], dtype=np.float32)
    # Scale the input using the fitted scaler
    input_scaled = scaler.transform(input_array)

    # Run TFLite inference
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

@app.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Fetches sensor data, runs inference, and renders an HTML page displaying:
    - Sensor values
    - Raw model output and predicted crop
    """
    try:
        response = requests.get(FIREBASE_SENSOR_URL)
    except Exception as e:
        return f"Error fetching sensor data: {e}", 500

    if response.status_code != 200:
        return "Error fetching sensor data from Firebase", 500
    sensor_data = response.json()
    if not sensor_data:
        return "No sensor data found in Firebase", 400

    try:
        temperature = float(sensor_data.get("temperature", 25.0))
        ph = float(sensor_data.get("pH", 7.0))
        humidity = float(sensor_data.get("humidity", 50.0))
        soil_moisture = float(sensor_data.get("soilMoisture", 0))
        sunlight_exposure = float(sensor_data.get("lux", 100))
    except ValueError as e:
        return f"Invalid sensor value: {e}", 400

    # Prepare input for TFLite model
    input_array = np.array([[temperature, ph, humidity, soil_moisture, sunlight_exposure]], dtype=np.float32)
    input_scaled = scaler.transform(input_array)
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = int(np.argmax(output_data[0]))
    predicted_label = label_mapping.get(predicted_class, "Unknown")

    # HTML template to display sensor data and model output
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
                                  temperature=temperature,
                                  ph=ph,
                                  humidity=humidity,
                                  soil_moisture=soil_moisture,
                                  sunlight_exposure=sunlight_exposure,
                                  output_data=output_data.tolist(),
                                  predicted_class=predicted_class,
                                  predicted_label=predicted_label)

if __name__ == "__main__":
    # Use the port provided by the environment or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
