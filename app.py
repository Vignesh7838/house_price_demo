import gradio as gr
import pickle
import numpy as np

# Load scaler and model
scaler = pickle.load(open('scaler.pkl', 'rb'))     # StandardScaler used during training
model = pickle.load(open('Linear.pkl', 'rb'))      # Trained LinearRegression model

def predict_price(square_footage, num_bedrooms, num_bathrooms, year_built, lot_size, garage_size, neighborhood_quality):

    # Prepare input as 2D array
    input_data = np.array([[
        square_footage,
        num_bedrooms,
        num_bathrooms,
        year_built,
        lot_size,
        garage_size,
        neighborhood_quality
    ]])
    
    # Scale input
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Format with dollar sign
    return f"${prediction:,.2f}"

demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Square Footage"),
        gr.Number(label="Number of Bedrooms"),
        gr.Number(label="Number of Bathrooms"),
        gr.Number(label="Year Built"),
        gr.Number(label="Lot Size"),
        gr.Number(label="Garage Size"),
        gr.Number(label="Neighborhood Quality")
    ],
    outputs=gr.Textbox(label="Predicted Price"),
    title="House Price Predictor",
    description="Enter house features to estimate the price using a scaled Linear Regression model."
)

demo.launch(share=True)




