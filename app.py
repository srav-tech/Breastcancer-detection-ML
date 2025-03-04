from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
try:
    model = pickle.load(open("breast_cancer_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print("❌ Error loading model or scaler:", e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict.html")
def predict_page():
    return render_template("predict.html")



# 🏥 Add routes for other static pages
@app.route("/prevention.html")
def prevention_page():
    return render_template("prevention.html")

@app.route("/causes.html")
def causes_page():
    return render_template("causes.html")

@app.route("/hospitals.html")
def hospitals_page():
    return render_template("hospitals.html")

@app.route("/research.html")
def research_page():
    return render_template("research.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("🔍 Form Data Received:", request.form)

        # Ensure all form fields are received
        if not all(k in request.form for k in ["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]):
            print("❌ Missing form values!")
            return render_template("predict.html", prediction_text="Error: Missing form values!")

        # Convert form inputs to float
        radius_mean = float(request.form["radius_mean"])
        texture_mean = float(request.form["texture_mean"])
        perimeter_mean = float(request.form["perimeter_mean"])
        area_mean = float(request.form["area_mean"])

        # Debugging: Print parsed values
        print(f"📊 Inputs: Radius={radius_mean}, Texture={texture_mean}, Perimeter={perimeter_mean}, Area={area_mean}")

        # Create a 30-feature input array with default values (you can replace 0 with dataset mean values)
        input_features = np.zeros(30)  # Default 30 features set to 0

        # Assign the user-provided values to the correct feature positions
        input_features[0] = radius_mean  # Mean Radius
        input_features[1] = texture_mean  # Mean Texture
        input_features[2] = perimeter_mean  # Mean Perimeter
        input_features[3] = area_mean  # Mean Area

        # Reshape and scale the input if the model requires it
        input_features = input_features.reshape(1, -1)

        if hasattr(scaler, "transform"):
            input_features = scaler.transform(input_features)

        # Prediction
        prediction = model.predict(input_features)
        result = "🔴 Malignant (Cancer Detected)" if prediction[0] == 1 else "🟢 Benign (No Cancer Detected)"

        print(f"✅ Prediction Result: {result}")

        return render_template("predict.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        print("❌ Error:", e)
        return render_template("predict.html", prediction_text="Error: Please enter valid inputs.")

if __name__ == "__main__":
    app.run(debug=True)