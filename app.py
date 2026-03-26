import streamlit as st

st.title("Road Damage Detection System")
st.write("Testing deployment...")

uploaded = st.file_uploader("Upload image", type=['jpg'])

if uploaded:
    st.image(uploaded)
    st.success("Working!")
