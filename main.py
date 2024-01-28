import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import re

# Load the dataset 
df = pd.read_excel("merged.xlsx")

# Extract features and target variable
X = df[['Brand', 'Size', 'Status', 'Type']]  # Features
y = df['Price']  # Target variable

# Define a function to extract numeric values after "FR" in the 'Size' column
def extract_fr_number(size):
    match = re.search(r'FR (\d+)', str(size))
    if match:
        return int(match.group(1))
    else:
        return None

def extract_cm_number(size):
    match = re.search(r'(\d+) cm', str(size))
    if match:
        return int(match.group(1))
    else:
        return None

def string_size_to_number(size):
    if size == "XXS":
        return 34
    elif size == "XS":
        return 36
    elif size == "S":
        return 38
    elif size == "M":
        return 40
    elif size == "L":
        return 42
    elif size == "XL":
        return 44
    elif size == "XXL" or size == "2XL":
        return 47
    elif size == "XXXL":
        return 54
    elif size == "4XL":
        return 60
    elif size == "6XL":
        return 66
    elif size == "8XL":
        return 72
    elif size == "Universel":
        return 42
    
    return size

def status_to_number(status):
    if status == "Neuf avec étiquette":
        return 1
    if status == "Neuf sans étiquette":
        return 2
    if status == "Très bon état":
        return 3
    if status == "Bon état":
        return 4
    if status == "Satisfaisant":
        return 5
    else:
        return 0

# Apply the function to the 'Size' column to create a new 'Size_FR' column
X['Size_FR'] = X['Size'].apply(extract_cm_number).apply(extract_fr_number).apply(string_size_to_number)
X['Status'] = X['Status'].apply(status_to_number)

# Drop the original 'Size' column
X = X.drop('Size', axis=1)

# One-hot encode the 'Brand' and 'Status' columns
preprocessor = ColumnTransformer(
    transformers=[
        ('type', OneHotEncoder(), ['Type']),
        ('brand', OneHotEncoder(), ['Brand']),
        ('status', 'passthrough', ['Status']),
        ('size', 'passthrough', ['Size_FR'])
    ])

# Fit and transform on training data
X_transformed = preprocessor.fit_transform(X)

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_transformed_imputed = imputer.fit_transform(X_transformed)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_transformed_imputed, y, test_size=0.4, random_state=34)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Prediction function
def predict_price(type, brand, size, status):
    input_data = pd.DataFrame({'Type': [type], 'Brand': [brand], 'Size': [size], 'Status': [int(status)]})

    input_data['Size_FR'] = input_data['Size'].apply(string_size_to_number)
    input_data = input_data.drop('Size', axis=1)
    
    input_data_transformed = preprocessor.transform(input_data)
    input_data_transformed_imputed = imputer.transform(input_data_transformed)
    prediction = model.predict(input_data_transformed_imputed)

    return prediction[0]

# HTTP Server
class PredictionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        type_input = data['type']
        brand_input = data['brand']
        size_input = data['size']
        status_input = data['status']

        predicted_price = predict_price(type_input, brand_input, size_input, status_input)

        response = {'predicted_price': predicted_price}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=PredictionHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
