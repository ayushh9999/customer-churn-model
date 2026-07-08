# Customer Churn Prediction App

An interactive Streamlit app that predicts whether a banking customer is likely to churn based on their profile and account details. The app uses a pre-trained Random Forest model and a saved scaler to transform user input and generate predictions in real time.

## Live Project Overview

This project is designed for quick churn risk checks. Users enter customer information in the sidebar, and the app returns:

- A churn prediction outcome
- Prediction probability
- The processed feature table used by the model
- The scaled values passed into the classifier

## Features

- Clean Streamlit interface
- Sidebar form for customer inputs
- Automatic feature engineering to match training data
- Model and scaler loaded from `.joblib` files
- Instant prediction with confidence scores
- Debug output for raw and scaled input data

## Files in This Project

- `app.py` - Streamlit application code
- `random_forest_model.joblib` - Trained churn prediction model
- `scaler.joblib` - Fitted scaler used before prediction
- `requirements.txt` - Python dependencies

## Input Fields Used By The App

The model expects these customer details:

- Credit Score
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Gender
- Geography

The app also creates extra engineered features such as age groups, tenure groups, balance-to-salary ratio, and one-hot encoded geography fields.

## Prerequisites

- Python 3.12 or newer
- pip
- A virtual environment is recommended

## Installation

Clone the repository and move into the project folder:

```powershell
git clone https://github.com/ayushh9999/customer-churn-model.git
cd customer-churn-model
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Run The App

Start the Streamlit app with:

```powershell
streamlit run app.py
```

The app will open in your browser automatically, or you can use the local URL shown in the terminal.

## How To Use

1. Open the app in your browser.
2. Enter the customer details in the sidebar.
3. Click **Predict Churn**.
4. Review the result and probability shown on the page.

## Model Notes

- The app loads `random_forest_model.joblib` and `scaler.joblib` at startup.
- If either file is missing, the app will stop and show an error message.
- Input features must stay in the same order as the training pipeline.

## Suggested Repository Structure

```text
customer-churn-model/
├─ app.py
├─ random_forest_model.joblib
├─ scaler.joblib
├─ requirements.txt
└─ README.md
```

## Deploying To Streamlit Community Cloud

If you want to publish the app online:

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select this repository.
5. Set the main file path to `app.py`.
6. Deploy the app.

## Troubleshooting

- If the app says the model or scaler is missing, make sure both `.joblib` files are in the same folder as `app.py`.
- If dependencies fail to install, recreate the virtual environment and run `pip install -r requirements.txt` again.
- If Streamlit does not open automatically, copy the local URL from the terminal and paste it into your browser.

## License

Add a license here if you want to share the project publicly.