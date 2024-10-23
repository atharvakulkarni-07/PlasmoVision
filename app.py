import streamlit as st
import gdown
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Google Drive model download link
model_url = 'https://drive.google.com/file/d/16g0bP-TgwXkkVK7Dv1ravImAjYdmYPx_/view?usp=sharing'  # Replace with your file ID

model_file = 'plasmovision_model.h5'

# Download the model if it's not already downloaded
if not os.path.exists(model_file):
    with st.spinner("Downloading model..."):
        gdown.download(model_url, model_file, quiet=False)

# Load the model
model = load_model(model_file)


# Custom CSS for gradient header, dark background, and button styles
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to bottom, #7A1D2D, #1B5E20);
        color: #f5f5f5;
    }

    h1 {
        color: white;  /* Set color to white to ensure it's visible */
        text-align: center;
        font-size: 3rem;
    }
    .stButton > button {
        background-color: #28a745;
        color: white;
        font-size: large;
        border-radius: 10px;
    }
    .stImage {
        border: 2px solid #007bff;  /* Add border to images */
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("### Upload Blood Smear Image")
uploaded_file = st.sidebar.file_uploader("", type=["jpg", "png", "jpeg"])



# Information section at the top
st.markdown('<h1>🦠 PlamoVision: Malaria Diagnosis</h1>', unsafe_allow_html=True)

# Description after 'Upload a blood smear...'
st.markdown("""
<div style="text-align: center;">
    <h3 style="color: #ffffff;">Upload a blood smear image and let the AI classify it!</h3>
    <p style="color: #cccccc; text-align: center;">
    Malaria is a life-threatening disease that is transmitted to humans through the bites of infected mosquitoes. 
    Early detection and accurate diagnosis are crucial in reducing its mortality rate. Typically, malaria diagnosis is performed 
    manually by skilled technicians analyzing blood smears under a microscope, a time-consuming and error-prone task. 
    PlamoVision aims to automate this process using a deep learning model.
    </p>
</div>
""", unsafe_allow_html=True)



# Image preprocessing function
def preprocess_image(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize((224, 224))  # Resize to 224x224
    img = np.array(img) / 255.0    # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.success("✅ Image uploaded successfully!")
    st.header("Uploaded Image")

    # Resize image to make it smaller
    st.image(img, caption='Uploaded Blood Smear Image', width=300)

    # Add a delay before showing the result to observe spinner
    with st.spinner("🔍 Analyzing..."):
        time.sleep(3)  # Add a 3-second delay for the loading animation
        img_preprocessed = preprocess_image(img)
        prediction = model.predict(img_preprocessed)

    # Result display with colors and emoji
    if prediction[0][0] > 0.5:
        result = '🩸 Uninfected'
        color = 'green'
        st.success('Congratulations! You are safe!')
        st.balloons()
    else:
        result = '🦠 Parasitized'
        color = 'red'
        st.warning('Please Take Care! You are diagnosed with Malaria!')

    st.markdown(f"<h2 style='color:{color}; text-align: center;'>{result}</h2>", unsafe_allow_html=True)
    
    # Show balloons when the result is shown
    

# Additional features to improve the app UI
st.sidebar.markdown("---")  # Separator line
st.sidebar.markdown("ℹ️ **Instructions**: Upload a blood smear image in the supported formats (JPG, PNG, JPEG).")

# Tooltip for file uploader
st.sidebar.markdown("💡 **Tip**: Ensure the image is clear for accurate diagnosis.")

# Add contact information
st.sidebar.markdown("---")  # Separator line
st.sidebar.markdown("### Contact Information")
st.sidebar.markdown("""
If you have any questions or feedback, feel free to reach out:

- **Name:** Atharva Vijay Kulkarni
- **Email:** atharvakulkarni.official@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/atharva-kulkarni-3b13a3255/
""")
