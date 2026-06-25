import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Exam Score Predictor", page_icon="📊")

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.title("Student Exam Score Predictor")
st.write("Enter a student's details to predict their final exam score.")

hours_studied = st.slider("Hours Studied (per week)", 0, 50, 20)
attendance = st.slider("Attendance (%)", 0, 100, 80)
sleep_hours = st.slider("Sleep Hours (per night)", 0, 12, 7)
previous_scores = st.slider("Previous Scores", 0, 100, 70)

input_df = pd.DataFrame(
    [[hours_studied, attendance, sleep_hours, previous_scores]],
    columns=features,
)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.metric("Predicted Exam Score", f"{prediction:.1f}")

    chart_df = pd.DataFrame(
        {"Feature": features, "Value": input_df.iloc[0].values}
    )
    st.bar_chart(chart_df.set_index("Feature"))
