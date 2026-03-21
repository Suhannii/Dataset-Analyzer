"""Fully adaptive auto-generated insights — works for any dataset type."""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter


# ── Dataset type detection ────────────────────────────────────────────────────

def _detect_dataset_profile(df: pd.DataFrame) -> dict:
    """
    Analyse the dataframe and return a profile dict describing what it contains.
    This drives which insight sections are shown.
    """
    profile = {
        "text_cols": [],
        "sentiment_col": None,
        "topic_col": None,
        "numeric_cols": [],
        "datetime_cols": [],
        "categorical_cols": [],
        "id_cols": [],
        "is_nlp": False,
        "is_numeric_heavy": False,
        "is_timeseries": False,
    }

    # Numeric
    num_cols = df.select_dtypes(include="number").columns.tolist()
    profile["numeric_cols"] = [c for c in num_cols
                                if df[c].nunique() > 10]  # exclude binary flags
    profile["id_cols"] = [c for c in num_cols
                          if df[c].is_monotonic_increasing and df[c].nunique() > len(df) * 0.9]

    # Datetime
    profile["datetime_cols"] = df.select_dtypes(include="datetime").columns.tolist()
    if profile["datetime_cols"]:
        profile["is_timeseries"] = True

    # Text / categorical
    for col in df.select_dtypes(include=["object", "category"]).columns:
        sample = df[col].dropna().astype(str)
        avg_len = sample.str.len().mean()
        nunique = df[col].nunique()
        unique_ratio = nunique / max(len(df), 1)

        if avg_len > 40:
            profile["text_cols"].append(col)
        elif nunique <= 15 and avg_len < 25:
            # Check if it looks like sentiment
            sentiment_vals = {"positive", "negative", "neutral", "very positive",
                              "very negative", "mixed", "0", "1", "2", "spam", "ham"}
            top_vals = set(sample.str.lower().value_counts().head(5).index)
            if top_vals & sentiment_vals:
                if profile["sentiment_col"] is None:
                    profile["sentiment_col"] = col
                else:
                    profile["categorical_cols"].append(col)
            else:
                profile["categorical_cols"].append(col)
        elif nunique <= 50 and unique_ratio < 0.1:
            profile["categorical_cols"].append(col)

    # Topic col = low-cardinality categorical that is NOT sentiment
    for col in profile["categorical_cols"]:
        if col != profile["sentiment_col"] and 2 <= df[col].nunique() <= 50:
            profile["topic_col"] = col
            break

    # Dataset type flags
    profile["is_nlp"] = len(profile["text_cols"]) > 0
    profile["is_numeric_heavy"] = len(profile["numeric_cols"]) >= 3

    return profile


# ── Section: Overview (always shown) ─────────────────────────────────────────

def _section_overview(df: pd.DataFrame, profile: dict):
    st.markdown("## 📋 Dataset Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")
    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    c5.metric("Memory", f"{mem:.2f} MB")

    # Column type breakdown
    type_map = {
        "Numeric": len(profile["numeric_cols"]),
        "Text": len(profile["text_cols"]),
        "Categorical": len(profile["categorical_cols"]),
        "Datetime": len(profile["datetime_cols"]),
        "ID": len(profile["id_cols"]),
    }
    type_map = {k: v for k, v in type_map.items() if v > 0}

    col_a, col_b = st.columns([1, 2])
    with col_a:
        fig = px.pie(names=list(type_map.keys()), values=list(type_map.values()),
                     title="Column Types", hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=240, margin=dict(t=40, b=0, l=0, r=0), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Missing values per column
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            mdf = pd.DataFrame({"column": missing.index,
                                "missing_%": (missing.values / len(df) * 100).round(2)})
            fig = px.bar(mdf, x="column", y="missing_%", title="Missing % per Column",
                         color="missing_%", color_continuous_scale="Oranges")
            fig.update_layout(height=240, margin=dict(t=40, b=0, l=0, r=0),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values in any column.")


# ── Section: Key Findings (always shown) ─────────────────────────────────────

def _section_key_findings(df: pd.DataFrame, profile: dict):
    st.markdown("## 🔑 Key Findings")
    findings = []

    total = len(df)

    # Sentiment finding
    sc = profile["sentiment_col"]
    if sc:
        vc = df[sc].value_counts()
        top, top_pct = vc.index[0], vc.iloc[0] / total * 100
        findings.append(f"Dominant class in `{sc}`: **{top}** ({top_pct:.1f}% of rows).")
        if top_pct > 60:
            findings.append(f"⚠️ Class imbalance detected in `{sc}` — consider resampling for ML.")

    # Numeric findings
    for col in profile["numeric_cols"][:3]:
        skew = df[col].skew()
        if abs(skew) > 2:
            direction = "right (positive)" if skew > 0 else "left (negative)"
            findings.append(f"`{col}` is highly skewed {direction} (skew={skew:.2f}). "
                            "Consider log-transform for modeling.")

    # Correlation finding
    num_df = df[profile["numeric_cols"]].dropna()
    if len(profile["numeric_cols"]) >= 2 and len(num_df) > 10:
        corr = num_df.corr().abs()
        np_corr = corr.values
        np.fill_diagonal(np_corr, 0)
        idx = np.unravel_index(np_corr.argmax(), np_corr.shape)
        c1, c2 = corr.columns[idx[0]], corr.columns[idx[1]]
        val = corr.iloc[idx[0], idx[1]]
        if val > 0.5:
            findings.append(f"Strong correlation between `{c1}` and `{c2}` (r={val:.2f}).")

    # Text findings
    for tc in profile["text_cols"][:1]:
        texts = df[tc].dropna().astype(str)
        avg_words = texts.str.split().str.len().mean()
        findings.append(f"Average text length in `{tc}`: **{avg_words:.1f} words**.")
        top3 = [w for w, _ in Counter(" ".join(texts).lower().split()).most_common(3)]
        findings.append(f"Most frequent words in `{tc}`: **{', '.join(top3)}**.")

    # Categorical findings
    for col in profile["categorical_cols"][:2]:
        vc = df[col].value_counts()
        top_pct = vc.iloc[0] / total * 100
        if top_pct > 70:
            findings.append(f"⚠️ `{col}` is dominated by **{vc.index[0]}** ({top_pct:.1f}%).")

    # Duplicates
    dups = df.duplicated().sum()
    if dups > 0:
        findings.append(f"⚠️ **{dups:,}** duplicate rows detected ({dups/total*100:.1f}%).")
    else:
        findings.append("✅ No duplicate rows found.")

    if not findings:
        findings.append("Upload a dataset to see auto-generated findings here.")

    for i, f in enumerate(findings, 1):
        st.markdown(f"**{i}.** {f}")


# ── Section: Numeric Analysis ─────────────────────────────────────────────────

def _section_numeric(df: pd.DataFrame, profile: dict):
    if not profile["numeric_cols"]:
        return
    st.markdown("## 📊 Numeric Column Insights")

    num_df = df[profile["numeric_cols"]]
    desc = num_df.describe().T.round(3)
    desc["skew"] = num_df.skew().round(3)
    desc["missing_%"] = (df[profile["numeric_cols"]].isnull().sum() / len(df) * 100).round(2)
    st.dataframe(desc[["mean", "std", "min", "50%", "max", "skew", "missing_%"]],
                 use_container_width=True)

    # Distribution grid (up to 6 cols)
    cols_to_plot = profile["numeric_cols"][:6]
    n = len(cols_to_plot)
    grid_cols = st.columns(min(n, 3))
    for i, col in enumerate(cols_to_plot):
        with grid_cols[i % 3]:
            fig = px.histogram(df, x=col, nbins=40, title=col,
                               color_discrete_sequence=["#1f77b4"])
            fig.update_layout(height=200, margin=dict(t=30, b=0, l=0, r=0),
                              showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    if len(profile["numeric_cols"]) >= 2:
        corr = num_df.corr().round(2)
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(min(len(corr), 10), min(len(corr), 8)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.5, square=True)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)


# ── Section: Categorical Analysis ────────────────────────────────────────────

def _section_categorical(df: pd.DataFrame, profile: dict):
    all_cat = []
    if profile["sentiment_col"]:
        all_cat.append(profile["sentiment_col"])
    all_cat += profile["categorical_cols"]
    if not all_cat:
        return

    st.markdown("## 🏷️ Categorical Column Insights")

    for col in all_cat[:4]:
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "count"]
        vc["pct"] = (vc["count"] / len(df) * 100).round(1)

        c_a, c_b = st.columns([2, 1])
        with c_a:
            fig = px.bar(vc.head(15), x=col, y="count", text="pct",
                         title=f"Distribution: {col}",
                         color=col,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(height=280, margin=dict(t=40, b=0, l=0, r=0),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c_b:
            st.markdown(f"**{col}**")
            st.write(f"Unique values: **{df[col].nunique()}**")
            st.write(f"Most common: **{vc.iloc[0][col]}** ({vc.iloc[0]['pct']}%)")
            if vc.iloc[0]["pct"] > 60:
                st.warning("Imbalanced")
            else:
                st.success("Balanced")


# ── Section: Text / NLP ───────────────────────────────────────────────────────

def _section_text(df: pd.DataFrame, profile: dict):
    if not profile["text_cols"]:
        return
    st.markdown("## 📝 Text Column Insights")

    for text_col in profile["text_cols"][:2]:
        st.markdown(f"#### Column: `{text_col}`")
        texts = df[text_col].dropna().astype(str)
        wc = texts.str.split().str.len()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total texts", f"{len(texts):,}")
        c2.metric("Unique texts", f"{texts.nunique():,}")
        c3.metric("Avg words", f"{wc.mean():.1f}")
        c4.metric("Max words", int(wc.max()))

        col_a, col_b = st.columns(2)
        with col_a:
            sc = profile["sentiment_col"]
            if sc and sc in df.columns:
                tmp = pd.DataFrame({"word_count": wc.values, sc: df[sc].values})
                fig = px.box(tmp, x=sc, y="word_count", color=sc,
                             title="Word Count by Category",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            else:
                fig = px.histogram(pd.DataFrame({"word_count": wc}),
                                   x="word_count", nbins=40,
                                   title="Word Count Distribution",
                                   color_discrete_sequence=["#1f77b4"])
            fig.update_layout(height=280, margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            all_words = " ".join(texts).lower().split()
            top15 = Counter(all_words).most_common(15)
            wdf = pd.DataFrame(top15, columns=["word", "count"])
            fig = px.bar(wdf, x="count", y="word", orientation="h",
                         title="Top 15 Words",
                         color_discrete_sequence=["#ff7f0e"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              height=280, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)


# ── Section: Time Series ──────────────────────────────────────────────────────

def _section_timeseries(df: pd.DataFrame, profile: dict):
    if not profile["datetime_cols"] or not profile["numeric_cols"]:
        return
    st.markdown("## 📅 Time Series Insights")
    dt_col = profile["datetime_cols"][0]
    val_col = profile["numeric_cols"][0]
    ts = df[[dt_col, val_col]].dropna().sort_values(dt_col)
    fig = px.line(ts, x=dt_col, y=val_col,
                  title=f"{val_col} over time",
                  color_discrete_sequence=["#1f77b4"])
    fig.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)


# ── Section: Data Quality ─────────────────────────────────────────────────────

def _section_quality(df: pd.DataFrame):
    st.markdown("## 🔍 Data Quality")
    missing = df.isnull().sum()
    dups = df.duplicated().sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Missing cells", f"{int(missing.sum()):,}")
    c2.metric("Duplicate rows", f"{int(dups):,}")
    c3.metric("Complete rows", f"{int((~df.isnull().any(axis=1)).sum()):,}")

    missing_cols = missing[missing > 0].sort_values(ascending=False)
    if not missing_cols.empty:
        mdf = pd.DataFrame({
            "column": missing_cols.index,
            "missing": missing_cols.values,
            "pct": (missing_cols.values / len(df) * 100).round(2)
        })
        fig = px.bar(mdf, x="column", y="pct", text="pct",
                     title="Missing Value % per Column",
                     color="pct", color_continuous_scale="Reds")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(height=280, margin=dict(t=40, b=0, l=0, r=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values detected.")

    single_val = [c for c in df.columns if df[c].nunique() <= 1]
    if single_val:
        st.warning(f"Single-value columns (no analytical value): `{'`, `'.join(single_val)}`")


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame):
    profile = _detect_dataset_profile(df)

    _section_overview(df, profile)
    st.divider()

    _section_key_findings(df, profile)
    st.divider()

    if profile["is_numeric_heavy"]:
        _section_numeric(df, profile)
        st.divider()

    sent_or_cat = profile["sentiment_col"] or profile["categorical_cols"]
    if sent_or_cat:
        _section_categorical(df, profile)
        st.divider()

    if profile["is_nlp"]:
        _section_text(df, profile)
        st.divider()

    if profile["is_timeseries"]:
        _section_timeseries(df, profile)
        st.divider()

    _section_quality(df)
