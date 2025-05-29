import streamlit as st
from PIL import Image
from pathlib import Path

from tools import render_footer

st.set_page_config(layout="wide", page_title="Прогнозирование диабета", page_icon="🩺")


@st.cache_data
def load_image(path):
    image = Image.open(path)
    MAX_SIZE = (600, 400)
    image.thumbnail(MAX_SIZE)
    return image


img_path = Path.cwd() / "assets" / "main_page_image.jpg"
image = load_image(img_path)

st.title("👋 Добро пожаловать!")
st.image(image)

st.markdown("""
### Это приложение предсказывает вероятность наличия диабета по введённым медицинским показателям.

**Навигация:**
- Перейдите на вкладку **Диагностика**, чтобы ввести данные и получить прогноз.
- Узнайте, благодаря чему осуществляется прогнозирование на вкладке **О приложении**.
- Ознакомьтесь с информацией о разработчике на вкладке **Контакты**.
""")

render_footer()
