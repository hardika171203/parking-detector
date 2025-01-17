import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pickle

# Load model dan dictionary
model = load_model('car1.keras')  # Pastikan model sudah kompatibel dengan TensorFlow 2.x
with open('spot_dict.pickle', 'rb') as handle:
    spot_dict = pickle.load(handle)

# Class dictionary
class_dictionary = {0: 'empty', 1: 'occupied'}

# Fungsi prediksi
def make_prediction(image):
    img = image / 255.0  # Normalisasi gambar
    image = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch
    class_predicted = model.predict(image, verbose=0)  # Matikan output verbose
    inID = np.argmax(class_predicted[0])  # Ambil indeks prediksi tertinggi
    label = class_dictionary[inID]
    return label

# Fungsi untuk menandai tempat parkir
def predict_on_image(image, spot_dict, color=[0, 255, 0], alpha=0.5):
    new_image = np.copy(image)
    overlay = np.copy(image)
    cnt_empty = 0
    all_spots = 0
    for spot in spot_dict.keys():
        all_spots += 1
        (x1, y1, x2, y2) = spot
        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
        spot_img = image[y1:y2, x1:x2]
        spot_img = cv2.resize(spot_img, (48, 48))
        label = make_prediction(spot_img)
        if label == 'empty':
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
            cnt_empty += 1
    cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)
    cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return new_image

# Streamlit UI
st.title("Parking Spot Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    predicted_image = predict_on_image(image, spot_dict)
    st.image(predicted_image, caption='Processed Image', use_column_width=True)
