# 🚗 Insurance Claim Predictor App

Interactive web application for predicting Belgian motor insurance claim severity.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

## How It Works

1. **Load Artifacts**: Downloads model, preprocessor, and categories from [Hugging Face Hub](https://huggingface.co/charlesnanakwakye/belgian-mtpl-claim-severity)
2. **User Input**: Collect policy and vehicle information via web form
3. **Preprocess**: Transform inputs using the same pipeline from training
4. **Predict**: Generate claim severity estimate in EUR

## Features

- 🎯 Real-time predictions
- 📊 Input validation and error handling
- 🔄 Cached model loading for fast responses
- 📱 Responsive design (works on mobile)

## Requirements

- Python 3.10+
- Internet connection (to download model from HF Hub)
- See `requirements.txt` for dependencies

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

No secrets needed - model is public on Hugging Face.
