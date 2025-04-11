import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ydata_profiling import ProfileReport
import sweetviz as sv
import streamlit.components.v1 as components

st.set_page_config(page_title="AutoPurgeAI_Pro", layout="wide")

st.title("🧹 AutoPurgeAI_Pro - Smart Data Cleaner + EDA Visualizer")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload a CSV file for profiling and cleaning", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")

    # Visualizations
    st.subheader("📈 Data Visualizations")

    col1, col2 = st.columns(2)

    # Scatter Plot
    with col1:
        st.markdown("### 🔹 Scatter Plot")
        if len(df.select_dtypes(include=np.number).columns) >= 2:
            x_col = st.selectbox("X-axis", df.columns, index=0)
            y_col = st.selectbox("Y-axis", df.columns, index=1)
            fig1 = sns.scatterplot(data=df, x=x_col, y=y_col)
            st.pyplot(fig1.figure)
        else:
            st.warning("Scatter plot needs at least two numeric columns.")

    # Histogram
    with col2:
        st.markdown("### 🔸 Histogram")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            hist_col = st.selectbox("Select Column", numeric_cols)
            fig2 = plt.figure()
            sns.histplot(df[hist_col], kde=True, bins=30)
            st.pyplot(fig2)
        else:
            st.warning("No numeric columns for histogram.")

    # Pie Chart
    st.subheader("🥧 Pie Chart")
    cat_cols = df.select_dtypes(include="object").columns
    if not cat_cols.empty:
        cat_col = st.selectbox("Choose Categorical Column", cat_cols)
        pie_data = df[cat_col].value_counts()
        fig3, ax3 = plt.subplots()
        ax3.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
        ax3.axis("equal")
        st.pyplot(fig3)
    else:
        st.warning("No categorical column available for pie chart.")

    st.markdown("---")

    # Correlation Heatmap
    st.subheader("🔗 Correlation Heatmap")
    if not df.select_dtypes(include=np.number).empty:
        corr = df.select_dtypes(include=np.number).corr()
        fig4 = plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig4)
    else:
        st.warning("No numeric columns to calculate correlation.")

    st.markdown("---")

    # Profiling Tools
    st.subheader("🧠 Automated Data Profiling Reports")

    tool = st.radio("Choose Profiling Tool", ["YData Profiling", "Sweetviz"])

    if tool == "YData Profiling":
        if st.button("🔍 Generate YData Profile Report"):
            profile = ProfileReport(df, explorative=True)
            profile.to_file("ydata_report.html")
            with open("ydata_report.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=1000, scrolling=True)

    elif tool == "Sweetviz":
        if st.button("🔍 Generate Sweetviz Report"):
            report = sv.analyze(df)
            report.show_html("sweetviz_report.html")
            with open("sweetviz_report.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=1000, scrolling=True)
