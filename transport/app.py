from flask import Flask, render_template, request, session
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "super_secret_key"

# Load trained models
with open("transport_model.pkl", "rb") as f:
    model_transport = pickle.load(f)

with open("hotel_model.pkl", "rb") as f:
    model_hotel = pickle.load(f)

# Load encoders
with open("encoders.pkl", "rb") as f:
    le_location, le_transport, le_hotel = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    show_recommendations = False
    selected_location = session.get("selected_location", None)

    if request.method == "POST":
        selected_location = request.form.get("location")
        session["selected_location"] = selected_location

        if selected_location in le_location.classes_:
            encoded_location = le_location.transform([selected_location])[0]
            input_data = [[encoded_location]]

            # Get probability predictions
            transport_probs = model_transport.predict_proba(input_data)[0]
            hotel_probs = model_hotel.predict_proba(input_data)[0]

            # Get top 5 recommendations
            top_transport_indices = np.argsort(transport_probs)[-5:][::-1]
            top_hotel_indices = np.argsort(hotel_probs)[-5:][::-1]

            transport_modes = le_transport.inverse_transform(top_transport_indices)
            hotels = le_hotel.inverse_transform(top_hotel_indices)

            prediction = {"transport": transport_modes, "hotel": hotels}
            show_recommendations = True
        else:
            prediction = {"error": "Location not found in dataset."}

    return render_template("index.html", prediction=prediction, locations=le_location.classes_, selected_location=selected_location, show_recommendations=show_recommendations)

if __name__ == "__main__":
    app.run(debug=True)
