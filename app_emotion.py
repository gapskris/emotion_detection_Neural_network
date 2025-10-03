# %%
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model('model/best_model.h5')

# Emotion mapping
reverse_emotion_map = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# App title
st.title("🧠 Emotion Detection App")
st.title("😡🤢😱😊😐😔😲")
st.write("Upload an image and let the model predict the emotion.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    IMG_HEIGHT, IMG_WIDTH = 48, 48
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    pred_index = np.argmax(prediction[0])
    pred_label = reverse_emotion_map[pred_index]

    st.subheader(f"🎯 Predicted Emotion: **{pred_label}**")

    # Plot probabilities
    st.write("### Emotion Probabilities")
    fig, ax = plt.subplots()
    emotions = [reverse_emotion_map[i] for i in range(len(prediction[0]))]
    ax.bar(emotions, prediction[0], color='skyblue')
    ax.set_ylabel("Confidence")
    ax.set_ylim([0, 1])
    plt.xticks(rotation=45)
    st.pyplot(fig)




