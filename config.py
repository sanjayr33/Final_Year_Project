import streamlit as st

# PAGE CONFIG
CSS = open("assets/css/styles.css", 'r').read()

# ASSETS
BACKGROUND = "assets/images/bg772.png"
BANNER = "assets/images/banner3.png"
DEFAULT_IMAGE = "assets/images/default.webp"
SIDE_BANNER = "assets/images/side_2.png"
EMOJI = "assets/images/emo75.png"
CHART = "assets/images/1254.png"
#Prediction Model
MODEL_PATH = "model/model.h5"
BEFORE_AUG = "assets/images/before_aug.png"
AFTER_AUG = "assets/images/after_aug.png"

# TEAM MEMBERS PAGE
TEAM_MEMBERS = [
    {"name": "", "role": "", "links":["", ""]},
    {"name": "", "role": "", "links":["", ""]},
    {"name": "KARAN P", "role": "", "links":["", ""]},
    {"name": "SANJAY R", "role": "", "links":["/", ""]},
    {"name": "THARANEESHWAR P", "role": "", "links":["", ""]},
    
]