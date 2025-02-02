from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['bedrooms']
    val2 = request.form['bathrooms']
    val3 = request.form['size in square feet']
    location = request.form['location']

    # Location encoding
    location_mapping = {"Pune": 1, "Mumbai": 2, "Delhi": 3}  
    location_val = location_mapping.get(location, 0)

    arr = np.array([val1, val2, val3, location_val])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=int(pred))


if __name__ == '__main__':
    app.run(debug=True)
