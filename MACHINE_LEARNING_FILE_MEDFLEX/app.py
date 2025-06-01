import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

# Function to generate the dataset
@st.cache_data
def generate_dataset():
    symptoms = [
        "fever", "runny nose", "fungal infection", "stomach acid", 
        "diarrhea", "pain", "bacterial infection", "headache", 
        "muscle pain", "sneezing", "itchy eyes", "skin rash", 
        "heartburn", "nausea", "stomach cramps", "joint pain", 
        "inflammation", "sore throat", "cough", "respiratory infection"
    ]

    medicines = {
        "fever": "Paracetamol",
        "runny nose": "Cetirizine",
        "fungal infection": "Cetirizine",
        "stomach acid": "Aciloc",
        "diarrhea": "Lomotil",
        "pain": "Diclofenac",
        "bacterial infection": "Azithromycin",
        "headache": "Paracetamol",
        "muscle pain": "Paracetamol",
        "sneezing": "Cetirizine",
        "itchy eyes": "Cetirizine",
        "skin rash": "Cetirizine",
        "heartburn": "Aciloc",
        "nausea": "Aciloc",
        "stomach cramps": "Lomotil",
        "joint pain": "Diclofenac",
        "inflammation": "Diclofenac",
        "sore throat": "Azithromycin",
        "cough": "Azithromycin",
        "respiratory infection": "Azithromycin"
    }

    data = []
    for _ in range(5000):
        selected_symptoms = random.sample(symptoms, random.randint(1, 3))
        suggested_medicine = list({medicines[symptom] for symptom in selected_symptoms})
        row = [" ".join(selected_symptoms), ", ".join(suggested_medicine)]
        data.append(row)

    df = pd.DataFrame(data, columns=["Symptoms", "Medicine"])
    return df, symptoms, medicines

# Function to train the model
@st.cache_resource
def train_model(df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["Symptoms"])
    y = df["Medicine"]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    return model, vectorizer

def main():
    st.title("Medicine Recommendation System 💊")
    st.write("This application recommends medicines based on symptoms you provide.")

    # Load dataset and train model
    df, symptoms, medicines = generate_dataset()
    model, vectorizer = train_model(df)

    # Display a multiselect for symptoms
    st.header("Select Your Symptoms:")
    selected_symptoms = st.multiselect(
        "Choose from the list of symptoms:",
        options=symptoms
    )

    # Display the selected symptoms
    if selected_symptoms:
        st.subheader("Selected Symptoms:")
        st.write(", ".join(selected_symptoms))

    # Predict medicines
    if st.button("Recommend Medicines"):
        if not selected_symptoms:
            st.error("Please select at least one symptom.")
        else:
            symptoms_text = " ".join(selected_symptoms)
            symptoms_vector = vectorizer.transform([symptoms_text])
            predicted_medicine = model.predict(symptoms_vector)
            st.success(f"Recommended Medicine(s): {predicted_medicine[0]}")

# Ensure the Streamlit application runs
if __name__ == "__main__":
    main()
