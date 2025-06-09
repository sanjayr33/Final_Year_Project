import base64
import streamlit as st  # type: ignore
from config import *
from streamlit_pages.Home_page import home_page
from streamlit_pages.Prediction_page import predict_page
from streamlit_pages.Team_page import team_page
import time

# Dummy credentials
CREDENTIALS = {
    
    "622021243028": "17102003",
    "622021243049": "22112004",
    "622021243059": "31072004",
}

# Set page config
st.set_page_config(page_title="Alzheimer's Detection Systems", page_icon=":brain:")

# Load CSS from config
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# Background image setup
def set_background(image_path):
    with open(image_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background(BACKGROUND)

# Sidebar styling
st.sidebar.image(SIDE_BANNER)
st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] > div:first-child {
        flex-direction: column;
        align-items: center;
        justify-content: center; 
    }
    [data-testid="stSidebar"] img {
         border-radius: 50px;
         height: 150px;
         width: 150px;
         box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.5); 
         margin: 0px 0px 0px 50px;
     }
     </style>
    """,
    unsafe_allow_html=True
)

# --- Main application function ---
def main_app():
    st.sidebar.title("Alzheimer's Detection System")
    st.sidebar.markdown("Deep learning model to detect Alzheimer's Disease from MRI scans.")
    st.sidebar.markdown("---")

    selection = st.sidebar.selectbox("Select a page to navigate:", ["Home Page", "Detection Page", "About us"])

    with st.sidebar.expander("🛑 Disclaimer"):
        st.markdown("""
        ⚠️ *This Alzheimer’s Disease Detection System is intended for educational and research purposes only.*
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always seek the guidance of a qualified healthcare provider regarding a medical condition.
        """)

    with st.sidebar.expander("📞 Contact Us"):
        st.markdown("""
        **Have questions, feedback, or need support? Reach out to us!**

        📧 Emails:
        - [sanjay223050@gmail.com](mailto:sanjay223050@gmail.com)
        - [kp677253@gmail.com](mailto:kp677253@gmail.com)
        - [daranish619@gmail.com](mailto:daranish619@gmail.com)

        📍 Location: Namakkal, India
        _We’d love to hear from you!_
        """)

    logout = st.sidebar.button("🚪 Logout")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    if logout:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


    # Page navigation
    if selection == "Home Page":
        home_page()
    elif selection == "Detection Page":
        predict_page()
    elif selection == "About us":
        team_page()

# --- Login System ---
def login_page():
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stToolbar"] {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
            with st.spinner("Loading app..."):
                time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid credentials")

# --- Main logic ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Logout button
if st.session_state.logged_in:
    main_app()
else:
    login_page()
