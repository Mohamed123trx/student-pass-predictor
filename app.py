import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np


model = joblib.load("student_model.pkl")


st.title("🎓 Predict Student Result Based on Study Hours")


hours = st.number_input("📚 Enter number of study hours:", min_value=0.0, max_value=24.0, step=0.5)

# زر التوقع
if st.button("Predict"):
    prediction = model.predict([[hours]])[0]
    prob = model.predict_proba([[hours]])[0][1] * 100

    st.write(f"🧠 Probability of passing: `{prob:.2f}%`")
    
    if prediction == 1:
        st.success("✅ Pass")
    else:
        st.error("❌ Fail")

    hours_range = np.linspace(0, 7, 300).reshape(-1, 1)
    prob_curve = model.predict_proba(hours_range)[:, 1]
    
    fig, ax = plt.subplots()
    ax.plot(hours_range, prob_curve, color='red', label='Sigmoid Curve')
    ax.scatter([[hours]], model.predict_proba([[hours]])[:, 1], color='blue', label='Your Input')
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Probability of Pass")
    ax.set_title("Sigmoid Curve - Probability vs Study Hours")
    ax.grid()
    ax.legend()

    st.pyplot(fig)