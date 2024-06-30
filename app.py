import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route('/submit_form', methods=['GET','POST'])
def submit_form():
    if request.method == 'POST':
        # Get form data
        type_of_food = request.form['typeOfFood']
        number_of_guests = int(request.form['numberOfGuests'])
        event_type = request.form['eventType']
        quantity_of_food = float(request.form['quantityOfFood'])
        storage_conditions = request.form['storageConditions']
        purchase_history = request.form['purchaseHistory']
        seasonality = request.form['seasonality']
        preparation_method = request.form['preparationMethod']
        geographical_location = request.form['geographicalLocation']
        pricing = float(request.form['pricing'])

        # Prepare input features for the model
        input_features = np.array([[type_of_food, number_of_guests, event_type, quantity_of_food,
                                    storage_conditions, purchase_history, seasonality,
                                    preparation_method, geographical_location, pricing]])

        # Make prediction
        prediction = model.predict(input_features)

        # Render result
        return render_template('result.html', prediction=prediction[0])

    return render_template('join.html')

if __name__ == '__main__':
    app.run(debug=True)
