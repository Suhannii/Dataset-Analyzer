"""Data cleaning pipeline."""
import re
import pandas as pd
import streamlit as st
from utils.config import (
    MISSING_DROP_ROW_THRESHOLD,
    MISSING_DROP_COL_THRESHOLD,
    CATEGORY_UNIQUE_THRESHOLD,
)
from utils.helpers import df_memory_mb


# ── Text column detection ─────────────────────────────────────────────────────

def _detect_text_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify columns that contain free-form text (tweets, reviews, comments etc.)
    vs short categorical/label/ID columns.
    A column is considered 'text' if its average value length > 30 chars.
    """
    text_cols = []
    for col in df.select_dtypes(include="object").columns:
        avg_len = df[col].dropna().astype(str).str.len().mean()
        if avg_len > 30:
            text_cols.append(col)
    return text_cols


def _detect_label_columns(df: pd.DataFrame) -> list[str]:
    """Identify low-cardinality string columns (labels, categories)."""
    label_cols = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        avg_len = df[col].dropna().astype(str).str.len().mean()
        unique_ratio = df[col].nunique() / max(len(df), 1)
        if avg_len <= 30 and unique_ratio < 0.05:
            label_cols.append(col)
    return label_cols


# ── Text cleaning functions ───────────────────────────────────────────────────

def _fix_encoding(text: str) -> str:
    """Fix mojibake / encoding artifacts."""
    if not isinstance(text, str):
        return text
    replacements = {
        "â€™": "'", "â€œ": '"', "â€\x9d": '"', "â€˜": "'",
        "â€¦": "...", "â\x80\x94": "-", "â\x80\x93": "-",
        "Ã©": "e", "Ã¨": "e", "Ã ": "a", "Ã¢": "a",
        "Ã®": "i", "Ã´": "o", "Ã»": "u", "Ã§": "c",
        "\u00e2\u0080\u0099": "'", "\u00e2\u0080\u009c": '"',
        "\u00e2\u0080\u009d": '"', "\u00e2\u0080\u00a6": "...",
        "\xa0": " ", "\\n": " ", "\\t": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def clean_text(text: str, options: dict) -> str:
    """Apply selected NLP cleaning steps to a single text string."""
    if not isinstance(text, str) or not text.strip():
        return text

    if options.get("fix_encoding", True):
        from modules.data_loader import _ENCODING_MAP
        for bad, good in _ENCODING_MAP.items():
            text = text.replace(bad, good)

    if options.get("remove_urls", True):
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)
        text = re.sub(r"pic\.[a-z]+\.[a-z]+/\S*", "", text, flags=re.IGNORECASE)
        # Remove domain-suffix noise tokens (e.g. "pictwittercom", "picly")
        text = re.sub(r"\b[a-z0-9]*(?:com|org|net|edu|gov|ly|io|co)\b", "", text, flags=re.IGNORECASE)
        # Remove alphanumeric hash tokens (e.g. "ddq8wexjls")
        text = re.sub(r"\b(?=[a-zA-Z]*\d)(?=\d*[a-zA-Z])[a-zA-Z0-9]{5,}\b", "", text)

    if options.get("remove_mentions", True):
        text = re.sub(r"@\w+", "", text)

    if options.get("remove_hashtags", False):
        # Remove entire hashtag token, not just the # symbol
        text = re.sub(r"#\w+", "", text)

    if options.get("remove_numbers", False):
        text = re.sub(r"\b\d+\b", "", text)

    if options.get("remove_punctuation", False):
        text = re.sub(r"[^\w\s]", "", text)

    if options.get("lowercase", False):
        text = text.lower()

    if options.get("remove_stopwords", False):
        from modules.data_loader import _STOPWORDS
        words = text.split()
        words = [w for w in words if w.lower() not in _STOPWORDS]
        text = " ".join(words)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\s,.\-:;!?]+|[\s,.\-:;!?]+$", "", text).strip()
    return text


# ── Auto-clean pipeline ───────────────────────────────────────────────────────

def auto_clean(df: pd.DataFrame, text_options: dict) -> tuple[pd.DataFrame, dict]:
    """
    Smart cleaning pipeline:
    - Text columns: NLP cleaning (encoding fix, URL/mention removal, whitespace)
    - Label/category columns: strip + title-case standardisation
    - All columns: drop exact duplicates, drop fully-empty rows/cols
    - Does NOT remove rows based on text length (preserves all data rows)
    """
    report = {}
    original_shape = df.shape

    text_cols = _detect_text_columns(df)
    label_cols = _detect_label_columns(df)

    # 1. Clean text columns (NLP)
    enc_fixed = 0
    for col in text_cols:
        before = df[col].copy()
        df[col] = df[col].apply(lambda x: clean_text(x, text_options))
        enc_fixed += (df[col] != before).sum()
    report["text_cols_cleaned"] = text_cols
    report["text_cells_modified"] = int(enc_fixed)

    # 2. Standardise label/category columns
    for col in label_cols:
        df[col] = df[col].astype(str).str.strip().str.title()
        df[col] = df[col].replace("Nan", pd.NA)
    report["label_cols_standardised"] = label_cols

    # 3. Strip whitespace from remaining string columns
    for col in df.select_dtypes(include="object").columns:
        if col not in text_cols and col not in label_cols:
            df[col] = df[col].str.strip()

    # 4. Drop rows where the text column is empty/null after cleaning
    before_empty = len(df)
    for col in text_cols:
        df = df[df[col].notna() & (df[col].str.strip() != "")]
    report["empty_text_rows_removed"] = before_empty - len(df)

    # 5. Drop exact duplicate rows
    before_dup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before_dup - len(df)

    # 6. Drop fully-empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    df = df.drop(columns=empty_cols)
    report["empty_cols_dropped"] = empty_cols

    report["shape_before"] = original_shape
    report["shape_after"] = df.shape
    return df, report


def show_auto_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Streamlit UI for the smart text-aware auto-clean pipeline."""
    st.markdown("### 🤖 Smart Text Cleaner")
    st.caption(
        "Cleans **text columns only** (tweets, reviews, comments). "
        "Label/category columns are standardised separately. "
        "No data rows are removed unless the text is empty after cleaning."
    )

    # Detect columns
    text_cols = _detect_text_columns(df)
    label_cols = _detect_label_columns(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", df.shape[0])
    c2.metric("Text columns detected", len(text_cols))
    c3.metric("Label columns detected", len(label_cols))

    if text_cols:
        st.info(f"Text columns: `{'`, `'.join(text_cols)}`")
    if label_cols:
        st.info(f"Label/category columns: `{'`, `'.join(label_cols)}`")

    st.markdown("#### Text Cleaning Options")
    col_a, col_b = st.columns(2)
    with col_a:
        fix_enc = st.checkbox("Fix encoding artifacts (â€™ → ')", value=True, key="opt_enc")
        remove_urls = st.checkbox("Remove URLs", value=True, key="opt_url")
        remove_mentions = st.checkbox("Remove @mentions", value=True, key="opt_mention")
        remove_hashtags = st.checkbox("Remove #hashtags", value=False, key="opt_hash")
    with col_b:
        lowercase = st.checkbox("Lowercase text", value=False, key="opt_lower")
        remove_numbers = st.checkbox("Remove standalone numbers", value=False, key="opt_num")
        remove_punct = st.checkbox("Remove punctuation", value=False, key="opt_punct")
        remove_stopwords = st.checkbox("Remove stopwords", value=False, key="opt_stop")

    options = {
        "fix_encoding": fix_enc,
        "remove_urls": remove_urls,
        "remove_mentions": remove_mentions,
        "remove_hashtags": remove_hashtags,
        "lowercase": lowercase,
        "remove_numbers": remove_numbers,
        "remove_punctuation": remove_punct,
        "remove_stopwords": remove_stopwords,
    }

    # Live preview on first text column
    if text_cols:
        sample_raw = df[text_cols[0]].dropna().iloc[0] if len(df) > 0 else ""
        sample_cleaned = clean_text(str(sample_raw), options)
        with st.expander("Preview: before vs after on first row"):
            st.markdown(f"**Before:** {sample_raw}")
            st.markdown(f"**After:** {sample_cleaned}")

    if st.button("🚀 Apply Text Cleaning", key="run_auto_clean", type="primary"):
        if not text_cols:
            st.warning("No text columns detected to clean.")
            return df
        with st.spinner("Cleaning text columns..."):
            cleaned, report = auto_clean(df.copy(), options)

        st.success("✅ Text cleaning complete!")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Text cells modified", report["text_cells_modified"])
        r2.metric("Duplicates removed", report["duplicates_removed"])
        r3.metric("Empty text rows removed", report["empty_text_rows_removed"])
        r4.metric("Empty cols dropped", len(report["empty_cols_dropped"]))
        st.info(f"Shape: {report['shape_before']} → {report['shape_after']}")

        st.session_state["df"] = cleaned
        df = cleaned

    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("### 🔍 Missing Values")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0]

    if missing_df.empty:
        st.success("✅ No missing values found.")
        return df

    st.dataframe(missing_df, width='stretch')

    strategy = st.selectbox(
        "Select cleaning strategy",
        ["Drop rows with missing values", "Drop columns with >50% missing",
         "Mean imputation (numeric)", "Median imputation (numeric)",
         "Mode imputation (categorical)", "Forward fill", "Backward fill"],
        key="missing_strategy"
    )

    if st.button("Apply Missing Value Strategy", key="apply_missing"):
        before_shape = df.shape
        if strategy == "Drop rows with missing values":
            cols_ok = missing_pct[missing_pct < MISSING_DROP_ROW_THRESHOLD * 100].index.tolist()
            df = df.dropna(subset=cols_ok)
        elif strategy == "Drop columns with >50% missing":
            drop_cols = missing_pct[missing_pct > MISSING_DROP_COL_THRESHOLD * 100].index.tolist()
            df = df.drop(columns=drop_cols)
            st.info(f"Dropped columns: {drop_cols}")
        elif strategy == "Mean imputation (numeric)":
            num_cols = df.select_dtypes(include="number").columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif strategy == "Median imputation (numeric)":
            num_cols = df.select_dtypes(include="number").columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        elif strategy == "Mode imputation (categorical)":
            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        elif strategy == "Forward fill":
            df = df.ffill()
        elif strategy == "Backward fill":
            df = df.bfill()

        st.success(f"✅ Applied. Shape: {before_shape} → {df.shape}")
        st.session_state["df"] = df

    return df


def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("### 🔁 Duplicate Rows")
    dup_count = df.duplicated().sum()
    dup_pct = round(dup_count / len(df) * 100, 2)

    if dup_count == 0:
        st.success("✅ No duplicate rows found.")
        return df

    st.warning(f"⚠️ Found {dup_count} duplicate rows ({dup_pct}%)")

    with st.expander("View duplicate records"):
        st.dataframe(df[df.duplicated(keep=False)].head(50), width='stretch')

    keep = st.radio("Keep which occurrence?", ["first", "last", "none (remove all)"], key="dup_keep")
    if st.button("Remove Duplicates", key="apply_dup"):
        keep_val = False if keep == "none (remove all)" else keep
        df = df.drop_duplicates(keep=keep_val)
        st.success(f"✅ Removed duplicates. Rows remaining: {len(df)}")
        st.session_state["df"] = df

    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("### ⚙️ Data Type Optimization")
    mem_before = df_memory_mb(df)
    changes = []

    for col in df.select_dtypes(include="object").columns:
        # Try datetime conversion
        try:
            converted = pd.to_datetime(df[col], infer_datetime_format=True)
            df[col] = converted
            changes.append(f"`{col}`: object → datetime")
            continue
        except Exception:
            pass
        # Convert to category if low cardinality
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < CATEGORY_UNIQUE_THRESHOLD:
            df[col] = df[col].astype("category")
            changes.append(f"`{col}`: object → category")

    mem_after = df_memory_mb(df)
    reduction = round((1 - mem_after / mem_before) * 100, 1) if mem_before > 0 else 0

    if changes:
        st.info("\n".join(changes))
        st.success(f"✅ Memory: {mem_before} MB → {mem_after} MB ({reduction}% reduction)")
    else:
        st.success("✅ Data types already optimal.")

    return df


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("### 📐 Outlier Detection (IQR Method)")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("No numeric columns for outlier detection.")
        return df

    col = st.selectbox("Select column", num_cols, key="outlier_col")
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    st.write(f"Outliers in `{col}`: **{len(outliers)}** rows (bounds: [{lower:.2f}, {upper:.2f}])")

    action = st.radio("Action", ["Keep", "Remove outliers", "Cap at bounds"], key="outlier_action")
    if st.button("Apply Outlier Action", key="apply_outlier"):
        if action == "Remove outliers":
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            st.success(f"✅ Removed {len(outliers)} outlier rows.")
        elif action == "Cap at bounds":
            df[col] = df[col].clip(lower=lower, upper=upper)
            st.success(f"✅ Capped `{col}` at [{lower:.2f}, {upper:.2f}].")
        st.session_state["df"] = df

    return df

