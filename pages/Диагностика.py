import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from tools import render_footer

st.title("📊 Диагностика диабета")

st.write("""
    ### Введите данные в боковой панели слева и получите прогноз риска развития сахарного диабета.
    """)

@st.cache_data
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    return model

model_path = Path.cwd() / "model.pkl"
model = load_model(model_path)

st.sidebar.header("Введите данные:")

features = {
    'pregnancies': st.sidebar.slider('Количество беременностей', 0, 10, 1),
    'glucose': st.sidebar.slider('Глюкоза', 44.0, 199.0, 100.0),
    'blood_pressure': st.sidebar.slider('Давление', 24, 122, 80),
    'skin_thickness': st.sidebar.slider('Толщина кожной складки', 7, 99, 20),
    'insulin': st.sidebar.slider('Инсулин', 14, 846, 300),
    'bmi': st.sidebar.slider('Индекс массы тела (BMI)', 18.2, 67.1, 30.0),
    'diabetes_pedigree_function': st.sidebar.slider('Наследственный фактор (DPF)', 0.078, 2.42, 1.0),
    'age': st.sidebar.slider('Возраст', 20, 80, 40),
}

df = pd.DataFrame([features])
st.write("### Введённые данные")
st.write(df)

prob = model.predict_proba(df.values)[0, 1]
st.write("### Вероятность диабета:")

if prob < 0.3:
    st.success(f"✅ {prob:.2%} — Низкий риск")
elif prob < 0.7:
    st.warning(f"⚠️ {prob:.2%} — Средний риск. Рекомендуется контроль и профилактика.")
else:
    st.error(f"🔸 {prob:.2%} — Высокий риск. Рекомендуется госпитализация.")

render_footer()