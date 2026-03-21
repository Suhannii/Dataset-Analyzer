"""Helper utilities."""
import io
import pandas as pd


def bytes_to_mb(b: int) -> float:
    return round(b / (1024 ** 2), 2)


def df_memory_mb(df: pd.DataFrame) -> float:
    return bytes_to_mb(df.memory_usage(deep=True).sum())


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()

