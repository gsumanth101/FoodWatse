import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_form', methods=['GET', 'POST'])
def submit_form():
    if request.method == 'POST':

        type_of_food = request.form['typeOfFood']
        number_of_guests = request.form['numberOfGuests']
        event_type = request.form['eventType']
        quantity_of_food = request.form['quantityOfFood']
        storage_conditions = request.form['storageConditions']
        purchase_history = request.form['purchaseHistory']
        seasonality = request.form['seasonality']
        preparation_method = request.form['preparationMethod']
        geographical_location = request.form['geographicalLocation']
        pricing = request.form['pricing']

        input_features = np.array([[type_of_food, number_of_guests, event_type, quantity_of_food, storage_conditions,
                                    purchase_history, seasonality, preparation_method, geographical_location, pricing
                                    ]])


        prediction = model.predict(input_features)


        return render_template('result.html', prediction=prediction[0])

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
