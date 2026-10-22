# Plant Leaf Disease Detection using CNN (MobileNetV2)

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## Dataset
Download the PlantVillage dataset from Kaggle ("PlantVillage Dataset", colored images)
and arrange it as:
```
data/PlantVillage/
    Apple___Apple_scab/*.jpg
    Apple___healthy/*.jpg
    ... (one folder per class)
```

## Train
```bash
python train.py --data_dir data/PlantVillage --model cnn         # baseline
python train.py --data_dir data/PlantVillage --model mobilenet   # transfer learning
```
Outputs go to `outputs/`: model (.keras), class_names.json, training_curves.png,
confusion_matrix.png, classification_report.txt.
Tip: with no GPU, use Google Colab or add `--epochs 3 --fine_tune_epochs 2`.

## Run the app
```bash
streamlit run app.py
```
To use the baseline model: `set MODEL_PATH=outputs/plant_disease_cnn.keras` first.
Note: the saved models include their own preprocessing, so the app feeds raw 0-255 pixels.

## Structure
train.py (data, models, evaluation) | app.py (Streamlit demo) | requirements.txt
