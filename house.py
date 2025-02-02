import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data_path = r'_YOUR_LOCATION_/house_price_data.csv'
data = pd.read_csv(data_path)

# Select relevant columns
selected_columns = ['bedrooms', 'bathrooms', 'size in square feet', 'location', 'price']
data = data[selected_columns]

# Split features and target variable
features = data.iloc[:, :4]
target = data.iloc[:, -1:]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
