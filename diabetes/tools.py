import streamlit as st

def render_footer():
    st.markdown(
        "<div style='height:40px;'></div>"
        "<p style='font-size:0.8em; color:gray; text-align:center;'>"
        "⚠️ Результат не является конечным диагнозом. Точный диагноз должен ставить врач на основе амбулаторных анализов."
        "</p>",
        unsafe_allow_html=True
    )