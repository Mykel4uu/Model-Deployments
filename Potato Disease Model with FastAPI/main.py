from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
import tensorflow as tf
from io import BytesIO
from PIL import Image


model = tf.keras.models.load_model(r'C:\Users\mine\PycharmProjects\AgriculturePotatoesDisease/model.h5')
classes = ["Early Blight", "Late Blight", "Healthy"]
app = FastAPI()


@app.get('/')
def hello():
    return {'message': 'Welcome to earth.'}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image as a numpy array
    image = read_file_as_image(await file.read())

    # Preprocess the image
    image = image.astype('float32') / 255.0
    image = tf.image.resize(image, [128, 128])
    image = np.expand_dims(image, axis=0)

    # Make the prediction using the loaded model
    prediction = model.predict(image)

    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_class_name = classes[predicted_class]
    confidence = np.max(prediction, axis=1)[0]

    return {'class': predicted_class_name,
            'confidence': float(confidence)}


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
