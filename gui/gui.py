import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
from PIL import Image, ImageTk

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Tkinter window
root = tk.Tk()
root.title("Heart Attack Predictor")
root.geometry("900x600")

# Load GIF background
bg_image = Image.open("gui/bg_gui.gif")
bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
bg_photo = ImageTk.PhotoImage(bg_image)


bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

title = tk.Label(root, text="❤️ Heart Attack Prediction System", 
                 font=("Arial", 22, "bold"), bg="white")
title.pack(pady=10)

# Input fields
labels = [
    "Age", "Smoker (0=No/1=Yes)", "Cigs Per Day", "BP Meds (0=NO /1 =YES)",
    "Hypertension (0 = NO/1= Yes)", "Diabetes (0=NO/1=YES)", "Total Cholesterol (125-200 mgdl)",
    "Systolic BP(80-130 mmHg)", "Diastolic BP (>80mmHg)", "BMI(18-24)", "Glucose(70-99 mg/dl)"
]

entries = []

form_frame = tk.Frame(root, bg="white")
form_frame.pack(pady=20)

for i, text in enumerate(labels):
    label = tk.Label(form_frame, text=text, font=("Arial", 12), bg="white")
    label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

    entry = tk.Entry(form_frame, font=("Arial", 12))
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

# Prediction function
def predict():
    try:
        values = [float(e.get()) for e in entries]

        values_scaled = scaler.transform([values])
        prediction = model.predict(values_scaled)[0]

        if prediction == 1:
            result = "⚠ High Risk of Heart Attack"
            color = "red"
        else:
            result = "✔ Low Risk of Heart Attack"
            color = "green"

        result_label.config(text=result, fg=color)

    except Exception as e:
        result_label.config(text="Error: Enter valid numbers!", fg="red")

# Predict Button
predict_button = tk.Button(
    root, text="Predict", font=("Arial", 14, "bold"),
    command=predict, bg="black", fg="white"
)
predict_button.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 18, "bold"), bg="white")
result_label.pack(pady=20)

root.mainloop()
