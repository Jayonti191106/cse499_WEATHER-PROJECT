from flask import Flask, request, render_template, redirect, url_for
import joblib

app = Flask(__name__, static_folder='static')

# Load the trained models for App 1 (Weather Prediction)
models = {}
for target in ['rainfall', 'humidity', 'sunshine', 'max_temp', 'min_temp']:
    model = joblib.load(f"models/{target.lower()}_model.pkl")
    models[target] = model

# List of all stations for App 1
stations_list = [
    "Ambagan(Ctg)", "Barisal", "Bhola", "Bogra", "Chandpur", "Chittagong", 
    "Chuadanga", "Comilla", "Cox's Bazar", "Dhaka", "Dinajpur", "Faridpur", 
    "Feni", "Hatiya", "Ishurdi", "Jessore", "Khepupara", "Khulna", "Kutubdia", 
    "Madaripur", "Mongla", "Mymensingh", "Patuakhali", "Rajshahi", "Rangamati", 
    "Rangpur", "Sandwip", "Satkhira", "Srimangal", "Sylhet", "Tangail", "Teknaf"
]

@app.route('/')
def homepage():
    return render_template('homepage.html')

# Routes for App 1 (Weather Prediction)
@app.route('/weather-prediction')
def weather_prediction():
    return render_template('index.html', stations=stations_list)

@app.route('/predict', methods=['POST'])
def predict():
    station = request.form['station']
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])

    station_label_encoded = convert_station_to_encoded(station)
    input_data = [[year, month, day, station_label_encoded]]

    predictions = {}
    for target, model in models.items():
        prediction = model.predict(input_data)[0]
        if target == 'rainfall':
            predictions[target] = f"{prediction:.2f} mm"
        elif target == 'humidity':
            predictions[target] = f"{prediction:.2f} %"
        elif target in ['max_temp', 'min_temp']:
            predictions[target] = f"{prediction:.2f} °C"
        else:
            predictions[target] = f"{prediction:.2f} Hours"

    return render_template('result.html', station=station, year=year, month=month, day=day, predictions=predictions)

# Routes for App 2 (Real-Time Weather)
@app.route('/realtime-weather')
def realtime_weather():
    return render_template('realtime_weather.html')

def convert_station_to_encoded(station):
    station_mapping = {
        'Ambagan(Ctg)': 0, 'Barisal': 1, 'Bhola': 2, 'Bogra': 3, 'Chandpur': 4,
        'Chittagong': 5, 'Chuadanga': 6, 'Comilla': 7, "Cox's Bazar": 8, 'Dhaka': 9,
        'Dinajpur': 10, 'Faridpur': 11, 'Feni': 12, 'Hatiya': 13, 'Ishurdi': 14,
        'Jessore': 15, 'Khepupara': 16, 'Khulna': 17, 'Kutubdia': 18, 'Madaripur': 19,
        'Mongla': 20, 'Mymensingh': 21, 'Patuakhali': 22, 'Rajshahi': 23, 'Rangamati': 24,
        'Rangpur': 25, 'Sandwip': 26, 'Satkhira': 27, 'Srimangal': 28, 'Sylhet': 29,
        'Tangail': 30, 'Teknaf': 31
    }
    return station_mapping.get(station, -1)

if __name__ == '__main__':
    app.run(debug=True)
