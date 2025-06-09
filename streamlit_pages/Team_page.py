import streamlit as st
from config import *


def team_page():
    st.markdown(f"<h1 style='text-align:center;'>Meet Our Dedicated Team Members</h1>", unsafe_allow_html=True)
    st.markdown("<br><br><div class='team-container'>", unsafe_allow_html=True)

    t_left, t_mid, t_right = st.columns(3)

    with t_mid:
        st.image(EMOJI)
        st.empty()

    b_left, b_mid, b_right = st.columns(3)

    with b_left:
        st.markdown(
            f"""
            <a href="{TEAM_MEMBERS[2]['links'][0]}">
                <div class='team-member'>
                    <h3>{TEAM_MEMBERS[2]['name']}</h3>
                    <p>{TEAM_MEMBERS[2]['role']}</p>
                </div>
            </a>
            <br>
            """,
            unsafe_allow_html=True
        )

    with b_mid:
        st.markdown(
            f"""
            <a href="{TEAM_MEMBERS[3]['links'][0]}">
                <div class='team-member'>
                    <h3>{TEAM_MEMBERS[3]['name']}</h3>
                    <p>{TEAM_MEMBERS[3]['role']}</p>
                </div>
            </a>
            <br>
            """,
            unsafe_allow_html=True
        )

    with b_right:
        st.markdown(
            f"""
            <a href="{TEAM_MEMBERS[4]['links'][0]}">
                <div class='team-member'>
                    <h3>{TEAM_MEMBERS[4]['name']}</h3>
                    <p>{TEAM_MEMBERS[4]['role']}</p>
                </div>
            </a>
            <br>
            """,
            unsafe_allow_html=True
        )