# ICU Mortality Risk Predictor

## Run locally
pip install -r requirements.txt
streamlit run app.py

All model artifacts must stay in the same folder as app.py.
The app asks for ~24 raw clinical inputs; other fields default to training medians.
Engineered features are computed automatically to match the training notebook.
