import random
import datetime
import numpy as np
import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pre-trained models for disease prediction and symptom checking
disease_model = load_model('models/disease_model.h5')
symptom_model = load_model('models/symptom_model.h5')

# Initialize data structures
symptom_list = ['fever', 'cough', 'headache', 'nausea', 'vomiting']
disease_list = ['common cold', 'flu', 'migraine', 'food poisoning', 'COVID-19']

# Vectorizer for user input
vectorizer = TfidfVectorizer()

# Sample medical database for appointments and medications
appointments = []
medications = {'Aspirin': 'Take one tablet every 4-6 hours', 'Paracetamol': 'Take one tablet every 6-8 hours'}

# Chatbot functions
def get_symptom_prediction(symptoms):
    # Dummy prediction for symptoms
    symptoms_vector = vectorizer.transform([symptoms])
    prediction = symptom_model.predict(symptoms_vector)
    return prediction

def get_disease_prediction(symptoms):
    # Dummy prediction for disease based on symptoms
    symptoms_vector = vectorizer.transform([symptoms])
    prediction = disease_model.predict(symptoms_vector)
    return disease_list[np.argmax(prediction)]

def schedule_appointment(date, time, doctor):
    appointment = {'date': date, 'time': time, 'doctor': doctor}
    appointments.append(appointment)
    return appointment

def get_medication_reminder(medication):
    return medications.get(medication, 'No information available')

# Sample user interactions
def chatbot_response(user_input):
    if 'symptom' in user_input:
        # Extract symptoms from user input
        symptoms = user_input.replace('symptom', '').strip()
        prediction = get_symptom_prediction(symptoms)
        return f'Based on your symptoms, you might have: {get_disease_prediction(symptoms)}'
    
    
    
    elif 'medication' in user_input:
        # Extract medication name from user input
        medication = user_input.replace('medication', '').strip()
        reminder = get_medication_reminder(medication)
        return f'Medication reminder: {reminder}'
    
    else:
        return 'I am sorry, I did not understand that. Can you please rephrase?'

