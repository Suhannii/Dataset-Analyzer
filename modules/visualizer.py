"""Chart generation — numeric, categorical, and NLP/text visualizations."""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from utils.config import MAX_HEATMAP_COLS, PAIRPLOT_SAMPLE_SIZE, MAX_PIE_CATEGORIES


# ── Column detection helpers ──────────────────────────────────────────────────

def _get_text_col(df: pd.DataFrame) -> str | None:
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        return None
    return max(obj_cols, key=lambda c: df[c].dropna().astype(str).str.len().mean())


def _get_sentiment_col(df: pd.DataFrame) -> str | None:
    """Prefer 'sentiment' column, then any low-cardinality string col."""
    for preferred in ["sentiment", "label_1", "label", "target", "class"]:
        if preferred in df.columns and 2 <= df[preferred].nunique() <= 10:
            return preferred
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if 2 <= df[col].nunique() <= 10:
            if df[col].dropna().astype(str).str.len().mean() < 20:
                return col
    return None


def _get_ngrams(series: pd.Series, n: int, top_k: int = 20) -> pd.DataFrame:
    tokens = " ".join(series.dropna().astype(str)).lower().split()
    grams = zip(*[tokens[i:] for i in range(n)])
    counts = Counter([" ".join(g) for g in grams])
    df_ng = pd.DataFrame(counts.most_common(top_k), columns=["ngram", "count"])
    return df_ng


# ── Standard visualizations ───────────────────────────────────────────────────

def plot_numeric(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("No numeric columns available.")
        return
    col = st.selectbox("Select numeric column", num_cols, key="viz_num_col")
    chart_type = st.selectbox("Chart type", ["Histogram + KDE", "Box Plot", "Scatter Plot"], key="viz_num_type")
    if chart_type == "Histogram + KDE":
        fig = px.histogram(df, x=col, marginal="violin", nbins=50,
                           title=f"Distribution of {col}", color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Box Plot":
        fig = px.box(df, y=col, title=f"Box Plot: {col}", color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Scatter Plot":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
            return
        col2 = st.selectbox("Y-axis", [c for c in num_cols if c != col], key="viz_scatter_y")
        fig = px.scatter(df, x=col, y=col2, title=f"{col} vs {col2}",
                         opacity=0.6, color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns.")
        return
    cols = num_cols[:MAX_HEATMAP_COLS]
    corr = df[cols].corr().round(2)
    fig, ax = plt.subplots(figsize=(min(len(cols), 12), min(len(cols), 10)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5, square=True)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)


def plot_categorical(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        st.info("No categorical columns available.")
        return
    col = st.selectbox("Select categorical column", cat_cols, key="viz_cat_col")
    chart_type = st.selectbox("Chart type", ["Bar Chart", "Pie Chart"], key="viz_cat_type")
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, "count"]
    if chart_type == "Bar Chart":
        fig = px.bar(vc.head(20), x=col, y="count", title=f"Value Counts: {col}",
                     color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        if df[col].nunique() > MAX_PIE_CATEGORIES:
            vc = vc.head(MAX_PIE_CATEGORIES)
        fig = px.pie(vc, names=col, values="count", title=f"Distribution: {col}")
        st.plotly_chart(fig, use_container_width=True)


def plot_datetime(df: pd.DataFrame):
    dt_cols = df.select_dtypes(include="datetime").columns.tolist()
    if not dt_cols:
        st.info("No datetime columns detected.")
        return
    dt_col = st.selectbox("Datetime column", dt_cols, key="viz_dt_col")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("Need a numeric column to plot over time.")
        return
    val_col = st.selectbox("Value column", num_cols, key="viz_dt_val")
    ts = df[[dt_col, val_col]].dropna().sort_values(dt_col)
    fig = px.line(ts, x=dt_col, y=val_col, title=f"{val_col} over time")
    st.plotly_chart(fig, use_container_width=True)


def plot_pairplot(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns.")
        return
    selected = st.multiselect("Select 2-4 numeric columns", num_cols,
                              default=num_cols[:min(3, len(num_cols))], key="viz_pair_cols")
    if len(selected) < 2:
        st.warning("Select at least 2 columns.")
        return
    sample = df[selected].dropna()
    if len(sample) > PAIRPLOT_SAMPLE_SIZE:
        sample = sample.sample(PAIRPLOT_SAMPLE_SIZE, random_state=42)
        st.info(f"Sampled {PAIRPLOT_SAMPLE_SIZE} rows for performance.")
    fig = px.scatter_matrix(sample, dimensions=selected, title="Pair Plot")
    st.plotly_chart(fig, use_container_width=True)


# ── NLP / Text visualizations ─────────────────────────────────────────────────

def plot_word_frequency(df: pd.DataFrame):
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        st.info("No text columns found.")
        return
    default = _get_text_col(df)
    text_col = st.selectbox("Text column", obj_cols, key="wf_col",
                            index=obj_cols.index(default) if default in obj_cols else 0)
    sent_col = _get_sentiment_col(df)
    top_n = st.slider("Top N words", 10, 50, 20, key="wf_topn")
    split_by = st.checkbox("Split by sentiment", value=bool(sent_col), key="wf_split")

    if split_by and sent_col:
        for lbl in sorted(df[sent_col].dropna().unique()):
            subset = df[df[sent_col] == lbl][text_col].dropna().astype(str)
            counts = Counter(" ".join(subset).lower().split()).most_common(top_n)
            wdf = pd.DataFrame(counts, columns=["word", "count"])
            fig = px.bar(wdf, x="count", y="word", orientation="h",
                         title=f"Top {top_n} words — {lbl}",
                         color_discrete_sequence=["#1f77b4"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
    else:
        counts = Counter(" ".join(df[text_col].dropna().astype(str)).lower().split()).most_common(top_n)
        wdf = pd.DataFrame(counts, columns=["word", "count"])
        fig = px.bar(wdf, x="count", y="word", orientation="h",
                     title=f"Top {top_n} Most Frequent Words",
                     color_discrete_sequence=["#1f77b4"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


def plot_wordcloud(df: pd.DataFrame):
    try:
        from wordcloud import WordCloud
    except ImportError:
        st.error("Run: `pip install wordcloud`")
        return
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        st.info("No text columns found.")
        return
    default = _get_text_col(df)
    text_col = st.selectbox("Text column", obj_cols, key="wc_col",
                            index=obj_cols.index(default) if default in obj_cols else 0)
    sent_col = _get_sentiment_col(df)
    filter_label = "All"
    if sent_col:
        options = ["All"] + sorted(df[sent_col].dropna().unique().tolist())
        filter_label = st.selectbox("Filter by sentiment", options, key="wc_label")

    subset = df if filter_label == "All" else df[df[sent_col] == filter_label]
    text = " ".join(subset[text_col].dropna().astype(str))
    if not text.strip():
        st.warning("No text to display.")
        return

    wc = WordCloud(width=900, height=450, background_color="white",
                   colormap="Blues", max_words=200).generate(text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud — {filter_label}", fontsize=14)
    st.pyplot(fig)
    plt.close(fig)


def plot_ngrams(df: pd.DataFrame):
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        st.info("No text columns found.")
        return
    default = _get_text_col(df)
    text_col = st.selectbox("Text column", obj_cols, key="ng_col",
                            index=obj_cols.index(default) if default in obj_cols else 0)
    n = st.radio("N-gram type", [2, 3],
                 format_func=lambda x: "Bigrams" if x == 2 else "Trigrams",
                 horizontal=True, key="ng_n")
    top_k = st.slider("Top K", 10, 30, 15, key="ng_topk")

    ng_df = _get_ngrams(df[text_col], n, top_k)
    fig = px.bar(ng_df, x="count", y="ngram", orientation="h",
                 title=f"Top {top_k} {'Bigrams' if n == 2 else 'Trigrams'}",
                 color_discrete_sequence=["#2ca02c"])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def plot_text_length(df: pd.DataFrame):
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        st.info("No text columns found.")
        return
    default = _get_text_col(df)
    text_col = st.selectbox("Text column", obj_cols, key="tl_col",
                            index=obj_cols.index(default) if default in obj_cols else 0)
    sent_col = _get_sentiment_col(df)
    metric = st.radio("Measure", ["Word count", "Character count"], horizontal=True, key="tl_metric")

    tmp = df[[text_col]].copy()
    if sent_col:
        tmp[sent_col] = df[sent_col]
    tmp["word_count"] = tmp[text_col].astype(str).str.split().str.len()
    tmp["char_count"] = tmp[text_col].astype(str).str.len()
    col_name = "word_count" if metric == "Word count" else "char_count"

    if sent_col and sent_col in tmp.columns:
        fig = px.box(tmp, x=sent_col, y=col_name, color=sent_col,
                     title=f"{metric} by Sentiment",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        fig = px.histogram(tmp, x=col_name, nbins=50, marginal="violin",
                           title=f"Distribution of {metric}",
                           color_discrete_sequence=["#1f77b4"])
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Avg {metric}", round(tmp[col_name].mean(), 1))
    c2.metric(f"Max {metric}", int(tmp[col_name].max()))
    c3.metric(f"Min {metric}", int(tmp[col_name].min()))


def plot_label_distribution(df: pd.DataFrame):
    sent_col = _get_sentiment_col(df)
    all_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not all_cat:
        st.info("No categorical columns found.")
        return
    default_idx = all_cat.index(sent_col) if sent_col and sent_col in all_cat else 0
    col = st.selectbox("Sentiment / Label column", all_cat, index=default_idx, key="ld_col")
    chart = st.radio("Chart type", ["Bar", "Pie", "Treemap"], horizontal=True, key="ld_chart")

    vc = df[col].value_counts().reset_index()
    vc.columns = [col, "count"]
    vc["pct"] = (vc["count"] / vc["count"].sum() * 100).round(2)

    if chart == "Bar":
        fig = px.bar(vc, x=col, y="count", text="pct",
                     title=f"Sentiment Distribution: {col}",
                     color=col, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
    elif chart == "Pie":
        fig = px.pie(vc, names=col, values="count",
                     title=f"Sentiment Distribution: {col}",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        fig = px.treemap(vc, path=[col], values="count",
                         title=f"Sentiment Treemap: {col}",
                         color="count", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(vc, use_container_width=True)


def plot_sentiment_vs_length(df: pd.DataFrame):
    text_col = _get_text_col(df)
    sent_col = _get_sentiment_col(df)
    if not text_col or not sent_col:
        st.info("Need both a text column and a sentiment column.")
        return
    tmp = df[[text_col, sent_col]].copy()
    tmp["word_count"] = tmp[text_col].astype(str).str.split().str.len()
    sample = tmp.sample(min(3000, len(tmp)), random_state=42)
    fig = px.box(sample, x=sent_col, y="word_count", color=sent_col,
                 title="Word Count Distribution by Sentiment",
                 points="outliers",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)


def plot_sentiment_over_topic(df: pd.DataFrame):
    """Stacked bar: sentiment breakdown per topic/game."""
    sent_col = _get_sentiment_col(df)
    if not sent_col:
        st.info("No sentiment column detected.")
        return

    # Find a topic column (low cardinality, not the sentiment col)
    topic_col = None
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col != sent_col and 2 <= df[col].nunique() <= 50:
            topic_col = col
            break

    if not topic_col:
        st.info("No topic/category column detected.")
        return

    grp = df.groupby([topic_col, sent_col]).size().reset_index(name="count")
    fig = px.bar(grp, x=topic_col, y="count", color=sent_col, barmode="stack",
                 title=f"Sentiment Breakdown by {topic_col}",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
