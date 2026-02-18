import streamlit as st
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from report import generate_pdf
from utils import setup_logger, load_data
from eda import render_kpis, render_gender_gap, render_distribution, render_correlation
from ml import train_model


st.set_page_config(
    page_title="Students Performance – ML Dashboard",
    layout="wide"
)

logger = setup_logger()

st.title("📊 Students Performance – ML Dashboard")
st.markdown("Author: Andrés Quirós Rojas")
st.markdown("---")


# =========================================================
# LOAD DATA
# =========================================================

try:
    df = load_data("StudentsPerformance.csv", logger)
except Exception:
    st.error("Dataset could not be loaded.")
    st.stop()


# =========================================================
# FILTERS
# =========================================================

st.sidebar.header("Filters")

gender = st.sidebar.selectbox(
    "Gender",
    ["All"] + list(df["gender"].unique())
)

if gender != "All":
    df = df[df["gender"] == gender]

if df.empty:
    st.warning("No data with selected filters.")
    st.stop()


# =========================================================
# EDA
# =========================================================

st.subheader("📌 KPIs")
render_kpis(df, st)

st.markdown("---")
st.subheader("👥 Gender Gap")
render_gender_gap(df, st)

st.markdown("---")
st.subheader("📈 Distribution")
render_distribution(df, st)

st.markdown("---")
st.subheader("🔎 Correlation")
render_correlation(df, st)


# =========================================================
# MACHINE LEARNING
# =========================================================

st.markdown("---")
st.header("🤖 Advanced Machine Learning")

try:

    (
        model,
        best_model_name,
        metrics,
        y_test,
        y_pred,
        feature_importance,
        shap_values
    ) = train_model(df, logger)

    st.subheader("📊 Model Performance (with Cross Validation)")

    for model_name, m in metrics.items():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{model_name} - R²", f"{m['r2']:.3f}")
        col2.metric(f"{model_name} - MAE", f"{m['mae']:.2f}")
        col3.metric(f"{model_name} - RMSE", f"{m['rmse']:.2f}")
        col4.metric(f"{model_name} - CV R²", f"{m['cv_mean']:.3f}")
        st.markdown("---")

    st.subheader(f"📉 Best Model: {best_model_name}")

    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "Actual", "y": "Predicted"},
        template="plotly_white",
        title="Actual vs Predicted"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    if feature_importance is not None:

        st.subheader("🔥 Feature Importance")

        fig_importance = px.bar(
            feature_importance.head(15),
            x="importance",
            y="feature",
            orientation="h",
            template="plotly_white"
        )

        st.plotly_chart(fig_importance, use_container_width=True)

    # =====================================================
    # SHAP
    # =====================================================

    if shap_values is not None:

        st.subheader("📊 SHAP Summary (Model Interpretability)")

        shap.summary_plot(shap_values, show=False)
        st.pyplot(bbox_inches="tight")
        plt.clf()


except Exception:
    st.error("Model training failed.")

# =====================================================
# DOWNLOAD REPORT
# =====================================================

st.markdown("---")
st.subheader("📄 Download ML Report")

pdf_buffer = generate_pdf(best_model_name, metrics)

st.download_button(
    label="Download PDF Report",
    data=pdf_buffer,
    file_name="students_performance_ml_report.pdf",
    mime="application/pdf"
)

# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.markdown("---")
st.header("📌 Business Insights")

st.markdown("""
### Key Findings:

- Reading and writing scores are the strongest predictors of math performance.
- Test preparation significantly improves outcomes.
- Socioeconomic indicators (lunch type) impact academic results.
- Feature importance and SHAP analysis confirm model interpretability.

This dashboard demonstrates advanced preprocessing pipelines, model comparison, cross validation, and explainable AI techniques.
""")
