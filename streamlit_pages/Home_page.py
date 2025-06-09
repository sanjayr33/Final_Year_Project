import streamlit as st
from config import BANNER



def home_page():
  st.image(BANNER)
  st.markdown("""
        ## 🧠 Introduction to Alzheimer's Disease
        **Alzheimer’s disease** is a progressive **neurological disorder** that primarily affects **memory, thinking, and behavior**. It is the most common cause of **dementia**, leading to a decline in **cognitive abilities** severe enough to interfere with daily life. The disease is characterized by the accumulation of **amyloid plaques** and **tau tangles** in the brain, which damage nerve cells. Early symptoms include **memory loss**, confusion, and difficulty completing familiar tasks. As the disease advances, patients may experience severe cognitive decline, **personality changes**, and **loss of independence**. While there is **no cure, treatments** and **lifestyle changes** can help manage symptoms and improve **quality of life**.
        """, unsafe_allow_html=True)
      
  st.markdown("---")    
  st.markdown("""  
        ## ⏰ Why Early Detection Matters
        Early detection of Alzheimer’s disease is crucial for timely intervention, allowing patients to receive treatment that may slow disease progression. It helps individuals and families plan for future care, financial management, and lifestyle adjustments. Early diagnosis enables participation in clinical trials and advanced therapies that may improve quality of life. It also provides an opportunity to adopt brain-healthy habits that can delay cognitive decline. Detecting Alzheimer’s early ensures better medical support and emotional preparedness for both patients and caregivers.
         """, unsafe_allow_html=True)
    
  st.markdown("---")
  st.markdown("""
        ## 🎯 Purpose of the project
        The purpose of this project is to develop an AI-powered web application for early detection of Alzheimer's disease using deep learning. It aims to assist doctors and researchers in diagnosing the disease more accurately and efficiently through MRI image analysis. By providing early Detection, the project helps in timely medical intervention and better patient management. Ultimately, it contributes to raising awareness and improving healthcare outcomes for Alzheimer’s patients.     
        <br>         
        """, unsafe_allow_html=True)
      
  st.markdown("---")
  st.subheader("📈 About the Model")
  st.markdown("""
   - Model: **EfficientNet** (deep learning architecture)
   - Dataset: **ADNI** (Alzheimer’s Disease Neuroimaging Initiative)
   - Frameworks: TensorFlow, Keras
   - Accuracy: **97% on validation set**
   - Loss: **0.12** (cross-entropy loss function)""")
      
  st.markdown("---")

  st.subheader("⚙️ How It Works")
  st.markdown("""
         1. **Upload** your brain MRI image.
         2. The **AI model** processes the scan.
         3. Get a **Detection result** instantly.
         4. Consult a doctor with the insight. """)
      
  st.markdown("---")
  st.subheader("🔍 Key Features")
  st.markdown("""
    - **AI-Based MRI Analysis**: Uses deep learning to analyze brain scans.
    - **Fast & Accurate**: Delivers Detection with up to **97% accuracy**.
    - **Private & Secure**: No data is stored or shared.
    - **Easy to Use**: Upload your MRI image and get results instantly.""")
  
  st.caption("Finished reading? Head over to the `Detection Page` to test the model with your MRI image.")