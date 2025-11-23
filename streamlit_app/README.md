# Insurance Claim Predictor App

A simple web app that predicts how much an insurance claim might cost. Just enter some details about the policy and vehicle, and it'll give you an estimate in euros.

## Getting Started

### Step 1: Train the Model First

Before you can use this app, you need to train the model. Go back to the main folder and run the `ml_belgium.ipynb` notebook. This will create the model files you need.

### Step 2: Install What You Need

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

This installs Streamlit and other tools the app needs.

### Step 3: Run the App

Still in the terminal, type:

```bash
streamlit run app.py
```

A browser window should open automatically showing the app. If it doesn't, go to `http://localhost:8501` in your browser.

## How to Use

1. Fill out the form with policy and vehicle information
2. Click the "Predict Claim Severity" button
3. See your prediction in euros!

That's it. Pretty simple, right?

## What the App Needs

The app looks for two files in the `best_model_backup` folder (one level up from here):
- `best_model.pkl` - The trained model
- `preprocessor.pkl` - The data preparation pipeline

If you see an error saying these files are missing, make sure you've run the notebook first.

## Want to Share It Online?

You can put this on Streamlit Cloud for free:

1. Push your code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign up
3. Connect your GitHub repo
4. Set the working directory to `streamlit_app`
5. Click deploy!

Now anyone with the link can use your app.
