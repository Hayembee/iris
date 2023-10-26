import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from PIL import Image

# Load the Iris dataset
iris = pd.read_csv('IRIS.csv')

# Mapping species to numerical values
iris['species'] = iris['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Separating features (X) and target variable (y)
X = iris.drop('species', axis=1)

with open('GradientBoostingClassifier.pkl', 'rb') as model_file:
   model = pickle.load(model_file)

st.title('Iris Flower Specie')
# Sidebar with input fields
st.sidebar.header('Input Parameters')
sepal_length = st.sidebar.slider('Sepal Length', float(X['sepal_length'].min()), float(X['sepal_length'].max()), float(X['sepal_length'].mean()))
sepal_width = st.sidebar.slider('Sepal Width', float(X['sepal_width'].min()), float(X['sepal_width'].max()), float(X['sepal_width'].mean()))
petal_length = st.sidebar.slider('Petal Length', float(X['petal_length'].min()), float(X['petal_length'].max()), float(X['petal_length'].mean()))
petal_width = st.sidebar.slider('Petal Width', float(X['petal_width'].min()), float(X['petal_width'].max()), float(X['petal_width'].mean()))

# Display the input parameters
st.sidebar.markdown('**Input Parameters:**')
st.sidebar.write(f'Sepal Length: {sepal_length}')
st.sidebar.write(f'Sepal Width: {sepal_width}')
st.sidebar.write(f'Petal Length: {petal_length}')
st.sidebar.write(f'Petal Width: {petal_width}')

# Predict button
if st.sidebar.button('Predict'):
    # Create a Pandas DataFrame with the input parameters
    input_data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })

    # Make predictions
    prediction = model.predict(input_data)[0]

    # Map the prediction to the corresponding species
    species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    predicted_species = species_mapping[prediction]

    # Display the predicted species
    st.write(f'Predicted Iris Species: {predicted_species}')

    # Dictionary to map species to image links
    image_links = {
        'Iris-setosa': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/220px-Irissetosa1.jpg',
        'Iris-versicolor': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Blue_Flag%2C_Ottawa.jpg/220px-Blue_Flag%2C_Ottawa.jpg',
        'Iris-virginica': 'https://www.fs.usda.gov/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica/iris_virginica_virginica.jpg'
    }

    # Display the image from the link
    try:
        image_link = image_links[predicted_species]
        st.image(image_link, caption=f'{predicted_species} Flower', use_column_width=True)
    except Exception as e:
        st.warning(f"Error loading image from the provided link: {e}")
