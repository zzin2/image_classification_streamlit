from PIL import Image
import pandas as pd
import plotly.express as px
import streamlit as st


def show_classification_result(img_f, model):
    img = Image.open(img_f)
    preds = model.predict(img)
    for i in preds:
        st.markdown(
            f"""
                <div style="
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 14px;
                    padding: 14px 16px;
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 14px; opacity: 0.7;"></div>
                    <div style="font-size: 18px; font-weight: 700; margin: 4px 0;">
                        {i['label']}
                    </div>
                    <div style="font-size: 14px;">
                        확률: <b>{i['score'] * 100:.2f}%</b>
                    </div>
                </div>
                """,
            unsafe_allow_html=True
        )
    return preds

def show_graph(model_predict: list):
    df = pd.DataFrame(model_predict)
    fig2 = px.bar(df, x="label", y="score", color="label")
    st.plotly_chart(fig2)