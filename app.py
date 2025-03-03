from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

# Dropdown mappings (English to Numeric)
fat_content_map = {'Regular': 1, 'Low Fat': 0}
outlet_size_map = {'Small': 2, 'Medium': 1, 'Large': 0}
location_type_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
outlet_type_map = {'SuperMarketType1': 1, 'SuperMarketType2': 2, 'GroceryStore': 0}
item_type_map = {
    'Baking Goods': 0, 'Breads': 1, 'Breakfast': 2, 'Canned': 3,
    'Dairy': 4, 'Frozen Foods': 5, 'Fruits & Vegetables': 6,
    'Hard Drinks': 7, 'Health & Hygiene': 8, 'Household': 9,
    'Meat': 10, 'Others': 11, 'Seafood': 12, 'Snack Foods': 13,
    'Soft Drinks': 14, 'Starchy Foods': 15
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Read form data
        item_mrp = float(request.form['item_mrp'])
        fat_content = fat_content_map[request.form['fat_content']]
        outlet_size = outlet_size_map[request.form['outlet_size']]
        location_type = location_type_map[request.form['location_type']]
        outlet_type = outlet_type_map[request.form['outlet_type']]
        item_type = item_type_map[request.form['item_type']]

        # Fixed values
        features = [
            779,            # Item_Identifier
            12.85,          # Item_Weight
            fat_content,    # Item_Fat_Content
            0.066,          # Item_Visibility
            item_mrp,       # Item_MRP
            5,              # Outlet_Identifier
            1997,           # Outlet_Establishment_Year
            outlet_size,    # Outlet_Size
            location_type,  # Outlet_Location_Type
            outlet_type,    # Outlet_Type
            item_type       # Item_Type
        ]

        # Predict using model
        prediction = model.predict(np.array(features).reshape(1, -1))

        return render_template('index.html', prediction=round(prediction[0], 2))

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
