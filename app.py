"""Main Streamlit application for Kaggle Dataset Analyzer."""
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Kaggle Dataset Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from modules.data_loader import validate_file, load_file, show_preview, fix_column_names, auto_clean_on_load
from modules.cleaner import clean_missing_values, clean_duplicates, optimize_dtypes, detect_outliers, show_auto_clean
from modules.analyzer import show_summary, show_column_analysis
from modules.visualizer import (
    plot_numeric, plot_correlation_heatmap,
    plot_categorical, plot_datetime, plot_pairplot,
    plot_word_frequency, plot_wordcloud, plot_ngrams,
    plot_text_length, plot_label_distribution,
    plot_sentiment_vs_length, plot_sentiment_over_topic,
)
from modules.insights import generate_insights
from utils.helpers import to_csv_bytes, to_excel_bytes

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Drop zone card */
.drop-zone {
    border: 2.5px dashed #1f77b4;
    border-radius: 16px;
    padding: 48px 32px;
    text-align: center;
    background: linear-gradient(135deg, #f0f6ff 0%, #ffffff 100%);
    transition: background 0.2s;
}
.drop-zone h2 { color: #1f77b4; margin-bottom: 8px; font-size: 1.6rem; }
.drop-zone p  { color: #555; margin: 4px 0; font-size: 0.95rem; }
.drop-zone .badge {
    display: inline-block;
    background: #e8f0fe;
    color: #1f77b4;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 4px;
    font-size: 0.85rem;
    font-weight: 600;
}
/* Hide default Streamlit file uploader label when inside drop zone */
.upload-inner [data-testid="stFileUploaderDropzone"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = None
if "original_df" not in st.session_state:
    st.session_state["original_df"] = None


def _process_upload(uploaded):
    """Load, fix columns, auto-clean and store in session state."""
    valid, err = validate_file(uploaded)
    if not valid:
        st.error(err)
        return
    with st.spinner("Loading and cleaning dataset..."):
        raw_df = load_file(uploaded)
        raw_df, was_fixed, new_names = fix_column_names(raw_df)
        st.session_state["original_df"] = raw_df.copy()
        cleaned_df, clean_report = auto_clean_on_load(raw_df.copy())
        st.session_state["df"] = cleaned_df
        st.session_state["loaded_file"] = uploaded.name
        st.session_state["clean_report"] = clean_report
        st.session_state["was_fixed"] = was_fixed
        st.session_state["new_names"] = new_names


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Kaggle Analyzer")
    st.caption("Smart data cleaning & insights")
    st.divider()

    # Compact sidebar uploader (always visible for re-upload)
    st.markdown("### 📁 Dataset")
    sidebar_file = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv", "xlsx", "xls"],
        help="Max 100 MB — drag & drop or click Browse",
        label_visibility="collapsed",
    )
    if sidebar_file and st.session_state.get("loaded_file") != sidebar_file.name:
        _process_upload(sidebar_file)

    # Show loaded file info
    if st.session_state.get("loaded_file"):
        df_loaded = st.session_state.get("df")
        st.success(f"✅ {st.session_state['loaded_file']}")
        if df_loaded is not None:
            st.caption(f"{df_loaded.shape[0]:,} rows · {df_loaded.shape[1]} cols")
        r = st.session_state.get("clean_report", {})
        if r.get("cells_fixed", 0) > 0:
            st.caption(f"🔧 {r['cells_fixed']:,} cells cleaned · "
                       f"{r['duplicates_removed']} dupes removed")
        if st.button("🗑️ Remove dataset", key="remove_ds"):
            for k in ["df", "original_df", "loaded_file", "clean_report",
                      "was_fixed", "new_names"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.divider()
    section = st.radio(
        "Navigate",
        ["📋 Preview", "🧹 Cleaning", "📊 Analysis", "📈 Visualizations",
         "💡 Insights", "💾 Download"],
        label_visibility="collapsed",
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔍 Kaggle Dataset Analyzer")
st.caption("Upload a dataset, clean it, explore insights, and download results.")

df: pd.DataFrame | None = st.session_state.get("df")

# ── Landing / Drop zone (shown when no file loaded) ───────────────────────────
if df is None:
    st.markdown("""
    <div class="drop-zone">
        <h2>📂 Drop your dataset here</h2>
        <p>Drag & drop a file onto the uploader below, or click <strong>Browse files</strong></p>
        <p style="margin-top:12px;">
            <span class="badge">CSV</span>
            <span class="badge">XLSX</span>
            <span class="badge">XLS</span>
        </p>
        <p style="margin-top:12px; color:#888;">Max file size: 100 MB</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="upload-inner">', unsafe_allow_html=True)
    center_file = st.file_uploader(
        "Drop file here or click to browse",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        key="center_uploader",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if center_file:
        _process_upload(center_file)
        st.rerun()

    st.stop()
    st.stop()

# ── Sections ──────────────────────────────────────────────────────────────────
if section == "📋 Preview":
    show_preview(df)

elif section == "🧹 Cleaning":
    st.header("🧹 Data Cleaning Pipeline")
    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        ["🤖 Text Cleaner", "Missing Values", "Duplicates", "Data Types", "Outliers"]
    )
    with tab0:
        df = show_auto_clean(df)
        st.session_state["df"] = df
    with tab1:
        df = clean_missing_values(df)
        st.session_state["df"] = df
    with tab2:
        df = clean_duplicates(df)
        st.session_state["df"] = df
    with tab3:
        df = optimize_dtypes(df)
        st.session_state["df"] = df
    with tab4:
        df = detect_outliers(df)
        st.session_state["df"] = df

elif section == "📊 Analysis":
    st.header("📊 Data Analysis")
    show_summary(df)
    st.divider()
    show_column_analysis(df)

elif section == "📈 Visualizations":
    st.header("📈 Visualizations")
    viz_tab = st.tabs([
        "Numeric", "Correlation", "Categorical", "Datetime", "Pair Plot",
        "📝 Word Frequency", "☁️ Word Cloud", "🔗 N-Grams",
        "📏 Text Length", "🏷️ Sentiment Distribution",
        "📊 Sentiment vs Length", "🎮 Sentiment by Topic",
    ])
    with viz_tab[0]:
        plot_numeric(df)
    with viz_tab[1]:
        plot_correlation_heatmap(df)
    with viz_tab[2]:
        plot_categorical(df)
    with viz_tab[3]:
        plot_datetime(df)
    with viz_tab[4]:
        plot_pairplot(df)
    with viz_tab[5]:
        plot_word_frequency(df)
    with viz_tab[6]:
        plot_wordcloud(df)
    with viz_tab[7]:
        plot_ngrams(df)
    with viz_tab[8]:
        plot_text_length(df)
    with viz_tab[9]:
        plot_label_distribution(df)
    with viz_tab[10]:
        plot_sentiment_vs_length(df)
    with viz_tab[11]:
        plot_sentiment_over_topic(df)

elif section == "💡 Insights":
    generate_insights(df)

elif section == "💾 Download":
    st.header("💾 Download Results")
    st.markdown("Download the cleaned dataset in your preferred format.")

    # Always use the latest session state df for download
    download_df = st.session_state.get("df", df)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ Download as CSV",
            data=to_csv_bytes(download_df),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            label="⬇️ Download as Excel",
            data=to_excel_bytes(download_df),
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.divider()
    st.markdown("### Before vs After")
    orig = st.session_state.get("original_df")
    if orig is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.write(f"Rows: {orig.shape[0]} | Columns: {orig.shape[1]}")
            st.write(f"Missing: {orig.isnull().sum().sum()} | Duplicates: {orig.duplicated().sum()}")
            st.dataframe(orig.head(5), width='stretch')
        with c2:
            st.markdown("**Cleaned**")
            st.write(f"Rows: {download_df.shape[0]} | Columns: {download_df.shape[1]}")
            st.write(f"Missing: {download_df.isnull().sum().sum()} | Duplicates: {download_df.duplicated().sum()}")
            st.dataframe(download_df.head(5), width='stretch')

