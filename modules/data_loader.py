"""File upload, validation, and loading with auto text cleaning."""
import io
import re
import chardet
import pandas as pd
import streamlit as st
from utils.config import MAX_FILE_SIZE_MB, LARGE_FILE_WARNING_MB, SUPPORTED_EXTENSIONS, PREVIEW_ROWS
from utils.helpers import bytes_to_mb


def validate_file(uploaded_file) -> tuple[bool, str]:
    name = uploaded_file.name.lower()
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported format '{ext}'. Please upload CSV or Excel files."
    size_mb = bytes_to_mb(uploaded_file.size)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File size ({size_mb} MB) exceeds the {MAX_FILE_SIZE_MB} limit."
    return True, ""


# ── Column name inference ─────────────────────────────────────────────────────

def _infer_col_name(series: pd.Series, idx: int) -> str:
    """Infer a semantic column name from the actual data values."""
    sample = series.dropna().astype(str).head(100)
    if len(sample) == 0:
        return f"col_{idx}"

    avg_len = sample.str.len().mean()

    # Long free-form text → tweet/review/comment
    if avg_len > 40:
        return "text"

    # Sentiment values → sentiment column
    sentiment_pattern = r"^(positive|negative|neutral|very positive|very negative|mixed)$"
    if sample.str.match(sentiment_pattern, case=False).mean() > 0.7:
        return "sentiment"

    # Binary label values
    binary_pattern = r"^(spam|ham|yes|no|true|false|0|1|2)$"
    if sample.str.match(binary_pattern, case=False).mean() > 0.7:
        return "label"

    # Check for date patterns
    if sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.5:
        return "date"

    # Purely numeric → likely an ID
    if sample.str.match(r"^\d+$").mean() > 0.9:
        return "id"

    # Low cardinality string → topic/game/category name
    unique_ratio = series.nunique() / max(len(series), 1)
    if unique_ratio < 0.02:
        return "topic"
    if unique_ratio < 0.10:
        return "category"

    return f"col_{idx}"


def _looks_like_no_header(df: pd.DataFrame) -> bool:
    """
    Return True if the column names look like data values, not header labels.
    Triggers when ≥50% of column names are: numeric, long text, or known label words.
    """
    hits = 0
    for col in df.columns:
        s = str(col).strip()
        if re.match(r"^\d+(\.\d+)?$", s):
            hits += 1
        elif len(s) > 30:
            hits += 1
        elif re.match(r"^(positive|negative|neutral|spam|ham|yes|no|true|false)$", s, re.I):
            hits += 1
        elif re.match(r"^[A-Z][a-zA-Z0-9 ]{2,}$", s) and " " not in s[:3]:
            # Proper-noun-like word (e.g. "Borderlands") — not a typical header
            hits += 0.5
    return (hits / max(len(df.columns), 1)) >= 0.5


def fix_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, list[str]]:
    """
    Detect headerless CSVs and assign inferred column names.
    Returns (df, was_fixed, names).
    """
    if not _looks_like_no_header(df):
        # Has a real header — just sanitize names
        names = []
        seen: dict[str, int] = {}
        for col in df.columns:
            n = re.sub(r"\s+", "_", str(col).strip()).lower()
            n = re.sub(r"[^\w]", "", n) or f"col_{len(names)}"
            if n in seen:
                seen[n] += 1
                n = f"{n}_{seen[n]}"
            else:
                seen[n] = 0
            names.append(n)
        df.columns = names
        return df, False, names

    # No header — restore the eaten first row, then infer names from data
    first_row = pd.DataFrame([df.columns.tolist()], columns=range(df.shape[1]))
    df.columns = range(df.shape[1])
    df = pd.concat([first_row, df], ignore_index=True)

    inferred = []
    used: dict[str, int] = {}
    for i in range(df.shape[1]):
        name = _infer_col_name(df.iloc[1:, i], i)  # skip restored first row for inference
        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 0
        inferred.append(name)

    df.columns = inferred

    # Fix mixed-type columns caused by prepending the string header row
    for col in df.columns:
        # Try to convert numeric-looking columns back to their proper type
        try:
            converted = pd.to_numeric(df[col])
            df[col] = converted
        except (ValueError, TypeError):
            df[col] = df[col].astype(str).replace("nan", pd.NA)

    return df, True, inferred


# ── Text cleaning (applied on load) ──────────────────────────────────────────

_ENCODING_MAP = {
    "â€™": "'", "â€œ": '"', "â€\x9d": '"', "â€˜": "'",
    "â€¦": "...", "â\x80\x94": "-", "â\x80\x93": "-",
    "Ã©": "e", "Ã¨": "e", "Ã ": "a", "Ã¢": "a",
    "Ã®": "i", "Ã´": "o", "Ã»": "u", "Ã§": "c",
    "\u00e2\u0080\u0099": "'", "\u00e2\u0080\u009c": '"',
    "\u00e2\u0080\u009d": '"', "\u00e2\u0080\u00a6": "...",
    "\xa0": " ", "\\n": " ", "\\t": " ",
}

# Load NLTK stopwords once at module level
try:
    from nltk.corpus import stopwords as _sw
    _STOPWORDS = set(_sw.words("english"))
except Exception:
    _STOPWORDS = {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "he","him","his","she","her","hers","it","its","they","them","their",
        "what","which","who","whom","this","that","these","those","am","is","are",
        "was","were","be","been","being","have","has","had","do","does","did",
        "a","an","the","and","but","if","or","because","as","until","while",
        "of","at","by","for","with","about","against","between","into","through",
        "during","before","after","above","below","to","from","up","down","in",
        "out","on","off","over","under","again","then","once","here","there",
        "when","where","why","how","all","both","each","few","more","most",
        "other","some","such","no","nor","not","only","own","same","so","than",
        "too","very","s","t","can","will","just","don","should","now","d","ll",
        "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn",
        "hasn","haven","isn","ma","mightn","mustn","needn","shan","shouldn",
        "wasn","weren","won","wouldn",
    }


def _clean_text_value(text: str) -> str:
    """Fix encoding, remove URLs/pic links/handles/hashtags, remove noise tokens, collapse whitespace."""
    if not isinstance(text, str):
        return text

    # Fix encoding artifacts first
    for bad, good in _ENCODING_MAP.items():
        text = text.replace(bad, good)

    # Remove full URLs (http/https/www) — must come before other patterns
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)

    # Remove pic.twitter.com/xxx and pic.instagram.com/xxx style media links
    text = re.sub(r"pic\.[a-z]+\.[a-z]+/\S*", "", text, flags=re.IGNORECASE)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove entire hashtag tokens (# + the concatenated word)
    text = re.sub(r"#\w+", "", text)
    # Remove leftover URL-fragment tokens: long alphanumeric strings that are
    # clearly not real words (contain digits mixed with letters, or known domain suffixes)
    # e.g. "ddq8wexjls", "Xk92mPz", "pictwittercom"
    # Rule: token is 8+ chars, has no vowel-consonant structure (all lowercase no spaces)
    # and contains digits OR is a known domain-like suffix
    text = re.sub(r"\b[a-z0-9]*(?:com|org|net|edu|gov|ly|io|co)\b", "", text, flags=re.IGNORECASE)
    # Remove tokens that mix letters and digits (hash-like noise)
    text = re.sub(r"\b(?=[a-zA-Z]*\d)(?=\d*[a-zA-Z])[a-zA-Z0-9]{5,}\b", "", text)

    # Remove stopwords
    words = text.split()
    words = [w for w in words if w.lower() not in _STOPWORDS]
    text = " ".join(words)

    # Collapse whitespace and clean up edges
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\s,.\-:;!?]+|[\s,.\-:;!?]+$", "", text).strip()

    return text


def auto_clean_on_load(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Automatically applied right after loading:
    - Fix encoding artifacts in text columns
    - Remove URLs and @mentions from text columns
    - Standardise label/category columns (strip + title-case)
    - Drop exact duplicate rows
    - Drop rows where text column is empty after cleaning
    Returns (cleaned_df, report).
    """
    report: dict = {}

    # Identify text vs label columns
    text_cols = [c for c in df.select_dtypes(include="object").columns
                 if df[c].dropna().astype(str).str.len().mean() > 40]
    label_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                  if c not in text_cols]

    # Clean text columns
    cells_fixed = 0
    for col in text_cols:
        before = df[col].copy()
        df[col] = df[col].apply(_clean_text_value)
        cells_fixed += int((df[col] != before).sum())
    report["text_cols"] = text_cols
    report["cells_fixed"] = cells_fixed

    # Standardise label columns
    for col in label_cols:
        df[col] = df[col].astype(str).str.strip().str.title().replace("Nan", pd.NA)
    report["label_cols"] = label_cols

    # Drop rows where text is now empty
    before = len(df)
    for col in text_cols:
        df = df[df[col].notna() & (df[col].str.strip() != "")]
    report["empty_text_rows_removed"] = before - len(df)

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(df)

    return df, report


# ── File loading ──────────────────────────────────────────────────────────────

def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    size_mb = bytes_to_mb(len(raw))

    if size_mb > LARGE_FILE_WARNING_MB:
        st.warning(f"⚠️ Large file ({size_mb} MB). Processing may be slow.")

    if name.endswith(".csv"):
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    else:
        df = pd.read_excel(io.BytesIO(raw))

    return df


def show_preview(df: pd.DataFrame):
    st.subheader("📋 Data Preview (first 10 rows)")
    st.dataframe(df.head(PREVIEW_ROWS), width='stretch')
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    from utils.helpers import df_memory_mb
    col3.metric("Memory", f"{df_memory_mb(df)} MB")

