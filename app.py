import tensorflow as tf
import numpy as np
from PIL import Image

from calories import calories

model = tf.keras.models.load_model("model/food_model.keras")

class_names = [
    "apple_pie",
    "burger",
    "fried_rice",
    "ice_cream",
    "pizza",
    "samosa"
]

image_path = input("Enter image path: ")

img = Image.open(image_path).convert("RGB")
img = img.resize((224,224))

img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

print("Food Item:", predicted_class)
print("Estimated Calories:", calories[predicted_class], "kcal")