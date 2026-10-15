# Poultry-House Monitor — KNN Demonstration

Streamlit app that classifies environmental readings as **Normal** or **Uncomfortable** using temperature, humidity, and light level.

> **Important:** The dataset and labels are synthetic, generated using illustrative rules plus a small amount of noise. Metrics only describe this simulated dataset; they do not establish real poultry-welfare accuracy or safe operating thresholds.

## Inputs
- `Temperature_C`: temperature in °C
- `Humidity_pct`: relative humidity in %
- `Light_Level_lux`: light level in lux
- `Condition`: target label (`Normal` / `Uncomfortable`)

## Files
- `app.py`: Streamlit app
- `synthetic_poultry_house_data.csv`: 1,200 synthetic examples
- `train_model.py`: reproducible training and evaluation
- `poultry_knn_model.joblib`: fitted StandardScaler + KNN pipeline
- `evaluation_metrics.json`: saved metrics and confusion matrix
- `requirements.txt`: dependencies

## Method
Stratified 80/20 train/test split (`random_state=42`), standardized features, KNN classifier, and 5-fold cross-validation for choosing `n_neighbors` and `weights`.

Test accuracy: **89.58%**  
Macro precision: **89.43%**  
Macro recall: **89.07%**  
Macro F1: **89.24%**  
Selected parameters: `{'knn__n_neighbors': 11, 'knn__weights': 'distance'}`

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a public GitHub repository, e.g. `poultry-house-monitor`.
2. Upload all files to the repository root.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Click **Create app**, select your repository and `main` branch, and set entrypoint to `app.py`.
5. Click **Deploy**. Once launched, copy the URL provided by Streamlit.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## For real use
Collect real sensor readings labelled by a qualified poultry expert, document the labelling process and sensor calibration, test on separate farms/dates, and re-evaluate before making operational decisions.
