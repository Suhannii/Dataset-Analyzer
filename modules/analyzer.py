"""Statistical analysis and summary."""
import pandas as pd
import streamlit as st
from utils.helpers import df_memory_mb


def show_summary(df: pd.DataFrame):
    st.subheader("📊 Dataset Summary")
    mem = df_memory_mb(df)
    missing_pct = round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2)
    dup_count = df.duplicated().sum()

    dtypes = df.dtypes
    type_counts = {
        "Numeric": int((dtypes.apply(pd.api.types.is_numeric_dtype)).sum()),
        "Categorical": int((dtypes == "category").sum() + (dtypes == "object").sum()),
        "Datetime": int((dtypes.apply(pd.api.types.is_datetime64_any_dtype)).sum()),
    }

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Memory", f"{mem} MB")
    c4.metric("Missing %", f"{missing_pct}%")
    c5.metric("Duplicates", dup_count)

    st.markdown("**Data Type Distribution**")
    t1, t2, t3 = st.columns(3)
    t1.metric("🔢 Numeric", type_counts["Numeric"])
    t2.metric("🔤 Categorical", type_counts["Categorical"])
    t3.metric("📅 Datetime", type_counts["Datetime"])


def show_column_analysis(df: pd.DataFrame):
    st.subheader("🔬 Column-Level Analysis")
    for col in df.columns:
        with st.expander(f"📌 {col} ({df[col].dtype})"):
            st.write(f"Unique values: **{df[col].nunique()}**")
            if pd.api.types.is_numeric_dtype(df[col]):
                stats = df[col].describe()
                st.dataframe(stats.to_frame().T, width='stretch')
                try:
                    from scipy.stats import skew, kurtosis
                    sk = round(skew(df[col].dropna()), 3)
                    ku = round(kurtosis(df[col].dropna()), 3)
                    st.write(f"Skewness: **{sk}** | Kurtosis: **{ku}**")
                except Exception:
                    pass
            else:
                top = df[col].value_counts().head(5).reset_index()
                top.columns = ["value", "count"]
                st.write("Top 5 values:")
                st.dataframe(top, width='stretch')

