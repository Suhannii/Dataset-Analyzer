# Kaggle Dataset Analyzer

A Streamlit dashboard for uploading, cleaning, analyzing, and visualizing Kaggle datasets.

## Setup

```bash
cd kaggle_analyzer
pip install -r requirements.txt
streamlit run app.py
```

## Features

- Upload CSV / Excel files (up to 100 MB)
- Auto-cleaning: missing values, duplicates, data type optimization, outlier detection
- Summary statistics and column-level analysis
- Interactive charts: histograms, box plots, scatter, heatmap, pair plots, time series
- Auto-generated insights
- Download cleaned dataset as CSV or Excel

## Project Structure

```
kaggle_analyzer/
├── app.py                 # Main Streamlit app
├── modules/
│   ├── data_loader.py
│   ├── cleaner.py
│   ├── analyzer.py
│   ├── visualizer.py
│   └── insights.py
├── utils/
│   ├── helpers.py
│   └── config.py
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```
