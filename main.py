from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Load the trained ANN model
model = tf.keras.models.load_model('annModel.keras')

# Initialize FastAPI app
app = FastAPI()

# Define the input schema using Pydantic
class PredictionRequest(BaseModel):
    features: list[float]  # List of input features

# Define a route for predictions
@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        # Extract features from request
        input_features = np.array(request.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)
        
        # Convert prediction to a plain Python data type
        predicted_value = float(prediction[0, 0])

        # Return prediction as response
        return {"prediction": predicted_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "API is working. Use /predict to make predictions."}


