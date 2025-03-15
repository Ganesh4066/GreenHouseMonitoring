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

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# -------------------------------------------------
# File paths for preprocessing objects and dataset
# -------------------------------------------------
scaler_path = "scaler.pkl"
label_encoder_path = "label_encoder.pkl"
data_path = "Modified_Crop_Data_Cleaned.csv"  # Ensure this CSV is in your deployment

# Define features and target (6 features)
features = ["temperature", "ph", "humidity", "soil_moisture", "sunlight_exposure", "soil_type"]
target = "label"

# -------------------------------------------------
# Load or fit preprocessing objects (scaler & label encoder)
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

# Build label mapping from integer to crop name
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
# Firebase URL for Sensor Data (fallback if no data in request)
# -------------------------------------------------
FIREBASE_SENSOR_URL = "https://green-house-monitoring-2a06d-default-rtdb.firebaseio.com/Greenhouse/SensorData.json"

# -------------------------------------------------
# Helper: Get sensor values (from request JSON or Firebase)
# -------------------------------------------------
def get_sensor_values():
    # Force JSON parsing even if Content-Type isn’t set properly
    data = request.get_json(force=True, silent=True)
    if data and all(key in data for key in ["temperature", "pH", "humidity", "soilMoisture", "lux"]):
        try:
            temperature = float(data.get("temperature", 25.0))
            ph = float(data.get("pH", 7.0))
            humidity = float(data.get("humidity", 50.0))
            soil_moisture = float(data.get("soilMoisture", 0))
            sunlight_exposure = float(data.get("lux", 100))
            soil_type = float(data.get("soil_type", 0.0))
            logging.debug("Sensor data received from request.")
        except ValueError as e:
            raise Exception(f"Invalid sensor value in request data: {e}")
    else:
        logging.debug("No valid sensor data in request; fetching from Firebase.")
        try:
            response = requests.get(FIREBASE_SENSOR_URL)
        except Exception as e:
            raise Exception(f"Error fetching sensor data from Firebase: {e}")
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
            soil_type = 0.0  # default value
        except ValueError as e:
            raise Exception(f"Invalid sensor value from Firebase: {e}")
    
    return {
        "temperature": temperature,
        "ph": ph,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "sunlight_exposure": sunlight_exposure,
        "soil_type": soil_type
    }

# -------------------------------------------------
# Helper: Run TFLite inference given sensor values
# -------------------------------------------------
def run_inference(sensor_values):
    input_features = [
        sensor_values["temperature"],
        sensor_values["ph"],
        sensor_values["humidity"],
        sensor_values["soil_moisture"],
        sensor_values["sunlight_exposure"],
        sensor_values["soil_type"]
    ]
    input_array = np.array([input_features], dtype=np.float32)
    logging.debug(f"Input array shape: {input_array.shape}")  # Should be (1, 6)
    try:
        input_scaled = scaler.transform(input_array)
        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
    except Exception as e:
        raise Exception(f"TFLite inference failed: {e}")
    
    predicted_class = int(np.argmax(output_data[0]))
    predicted_label = label_mapping.get(predicted_class, "Unknown")
    return output_data, predicted_class, predicted_label

# -------------------------------------------------
# Flask Endpoints
# -------------------------------------------------
@app.route("/")
def home():
    return "Crop Prediction API using Firebase data is running."

# /predict endpoint supports GET and POST requests.
@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        sensor_values = get_sensor_values()
        output_data, predicted_class, predicted_label = run_inference(sensor_values)
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500

    message = (
        f"Predicted crop: '{predicted_label}'. Conditions: Temperature = {sensor_values['temperature']}°C, "
        f"pH = {sensor_values['ph']}, Humidity = {sensor_values['humidity']}%, "
        f"Soil Moisture = {sensor_values['soil_moisture']}, "
        f"Sunlight Exposure = {sensor_values['sunlight_exposure']} lux."
    )
    
    response_json = {
        "sensor_data": sensor_values,
        "predicted_class": predicted_class,
        "predicted_crop": predicted_label,
        "raw_model_output": output_data.tolist(),
        "message": message
    }
    return jsonify(response_json), 200

# /dashboard endpoint renders an improved UI using Bootstrap.
@app.route("/dashboard", methods=["GET"])
def dashboard():
    try:
        sensor_values = get_sensor_values()
        output_data, predicted_class, predicted_label = run_inference(sensor_values)
    except Exception as e:
        logging.error(e)
        return f"Error: {e}", 500

    message = (
        f"Predicted crop: '{predicted_label}'. Conditions: Temperature = {sensor_values['temperature']}°C, "
        f"pH = {sensor_values['ph']}, Humidity = {sensor_values['humidity']}%, "
        f"Soil Moisture = {sensor_values['soil_moisture']}, "
        f"Sunlight Exposure = {sensor_values['sunlight_exposure']} lux."
    )
    
    html = """
    <!doctype html>
    <html lang="en">
      <head>
         <meta charset="utf-8">
         <title>Greenhouse Monitoring Dashboard</title>
         <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
         <style>
             body { background-color: #f4f4f4; padding-top: 20px; }
             .card { margin-bottom: 20px; }
             pre { background-color: #e9ecef; padding: 10px; border-radius: 4px; }
         </style>
      </head>
      <body>
         <div class="container">
            <h1 class="text-center mb-4">Greenhouse Monitoring Dashboard</h1>
            <div class="row">
               <div class="col-md-6">
                  <div class="card">
                     <div class="card-header">Sensor Data</div>
                     <ul class="list-group list-group-flush">
                        <li class="list-group-item">Temperature: {{ temperature }} °C</li>
                        <li class="list-group-item">pH: {{ ph }}</li>
                        <li class="list-group-item">Humidity: {{ humidity }} %</li>
                        <li class="list-group-item">Soil Moisture: {{ soil_moisture }}</li>
                        <li class="list-group-item">Sunlight Exposure: {{ sunlight_exposure }} lux</li>
                        <li class="list-group-item">Soil Type: {{ soil_type }}</li>
                     </ul>
                  </div>
               </div>
               <div class="col-md-6">
                  <div class="card">
                     <div class="card-header">Model Output</div>
                     <ul class="list-group list-group-flush">
                        <li class="list-group-item">Predicted Crop: <strong>{{ predicted_label }}</strong></li>
                        <li class="list-group-item">Predicted Class: {{ predicted_class }}</li>
                        <li class="list-group-item">Raw Output: <pre>{{ output_data }}</pre></li>
                     </ul>
                  </div>
               </div>
            </div>
            <div class="alert alert-info text-center" role="alert">
                {{ message }}
            </div>
         </div>
         <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
         <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"></script>
      </body>
    </html>
    """
    return render_template_string(
        html,
        temperature=sensor_values["temperature"],
        ph=sensor_values["ph"],
        humidity=sensor_values["humidity"],
        soil_moisture=sensor_values["soil_moisture"],
        sunlight_exposure=sensor_values["sunlight_exposure"],
        soil_type=sensor_values["soil_type"],
        output_data=output_data.tolist(),
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        message=message
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
