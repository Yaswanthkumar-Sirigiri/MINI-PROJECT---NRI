# FreshGuard AI - Food Spoilage Prediction UI

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

Keep these files in the same folder as app.py:
spoilage_risk_model.pkl, shelf_life_model.pkl, risk_label_encoder.pkl, feature_columns.pkl

Note: scikit-learn in requirements.txt matches the version used to train the models.
If you retrain in Colab, use the Colab scikit-learn version here instead (check with: import sklearn; sklearn.__version__).
