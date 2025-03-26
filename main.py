import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    
    ### How It Works
    1. *Upload Image:* Go to the *Disease Recognition* page and upload an image of a plant with suspected diseases.
    2. *Analysis:* Our system will process the image using advanced algorithms to identify potential diseases.
    3. *Results:* View the results and recommendations for further action.
    
    ### Get Started
    Click on the *Disease Recognition* page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    # Check if the file is uploaded
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
    
    # Predict Button
    if st.button("Predict"):
        if test_image is not None:
            with st.spinner("Please Wait..."):
                # Display the uploaded image again after prediction
                st.image(test_image, caption="Uploaded Image", use_container_width=True)  # Image displayed even after prediction
                st.write("Our Prediction")
                result_index = model_prediction(test_image)
                
                # Define Class Names
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)_Powdery_mildew', 'Cherry_(including_sour)_healthy',
                    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)Common_rust',
                    'Corn_(maize)_Northern_Leaf_Blight', 'Corn_(maize)_healthy', 'Grape___Black_rot',
                    'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange__Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,bell__Bacterial_spot', 'Pepper,bell__healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]
                st.success(f"Model is Predicting: It's a {class_name[result_index]}")
        else:
            st.error("Please upload an image before clicking 'Predict'.")
