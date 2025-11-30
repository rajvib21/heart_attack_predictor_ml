Heart Attack Predictor
Machine Learning • Tkinter GUI • Offline Desktop App
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/ML-ScikitLearn-orange?logo=scikitlearn&logoColor=white" /> <img src="https://img.shields.io/badge/GUI-Tkinter-green" /> <img src="https://img.shields.io/badge/License-MIT-lightgrey" /> </p>

A desktop-based Heart Attack Risk Prediction System built using
Machine Learning (Decision Tree Classifier) and a Tkinter GUI with an animated background.

The model predicts whether a person is at High Risk or Low Risk based on medical inputs.


Project Structure
heartattackpredictor/
│
├── data/
│   └── heartattack.csv
│
├── models/
│   ├── dt_model.pkl
│   └── scaler.pkl
│
├── gui/
│   ├── gui.py
│   └── bg_gui.gif
│
├── analysis.py
├── requirements.txt
└── README.md

🚀 Features

✔️ Machine learning-powered prediction

✔️ Clean Tkinter GUI with GIF background

✔️ Offline desktop app (no internet required)

✔️ Easy to train & retrain

✔️ Beginner-friendly and well-structured

🛠️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/rajvib21/heart_attack_predictor_ml.git
cd heartattackpredictor

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ (Optional) Retrain the ML Model
python analysis.py


This creates:

models/dt_model.pkl
models/scaler.pkl

4️⃣ Run the GUI
python gui/gui.py

🧠 Model Details

The project uses:

Decision Tree Classifier

StandardScaler (for logistic model)

Trained on medical features such as:

age

smoker / cigs per day

blood pressure

cholesterol

diabetes

BMI

glucose

hypertension

The model outputs:

0 → Low Risk

1 → High Risk

🖥️ GUI Preview

The GUI includes:

Animated GIF background

Clean input fields

Single "Predict" button

Popup result window with prediction

📊 Dataset

The dataset (heartattack.csv) contains real medical records and parameters relevant to CHD (Coronary Heart Disease) prediction.

🔮 Future Improvements

Add charts & visual reports

Add database storage for patient history

Add dark mode GUI

Convert to a web app (Flask / FastAPI)

Add voice input

🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to modify.
