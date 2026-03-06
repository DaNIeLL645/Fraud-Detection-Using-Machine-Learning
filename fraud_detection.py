import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            transaction_type TEXT,
            amount REAL,
            oldbalanceOrg REAL,
            newbalanceOrig REAL,
            oldbalanceDest REAL,
            newbalanceDest REAL,
            prediction INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def save_transaction(trans_type, amt, old_org, new_org, old_dest, new_dest, pred):
    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO predictions_history 
        (timestamp, transaction_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, trans_type, amt, old_org, new_org, old_dest, new_dest, pred))
    
    conn.commit()
    conn.close()

init_db()

@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")

model = load_model()

st.title("Fraud Detection Prediction APP")
st.markdown("Please enter the transaction details")
st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])

amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=900.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [
            {
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest
            }
        ]
    )

    prediction = model.predict(input_data)[0]
    pred_value = int(prediction)
    
    st.subheader(f"Prediction: '{pred_value}'")

    if pred_value == 1:
        st.error("This transaction can be fraud")
    else:
        st.success("This transaction looks like it is not fraud")
        
    try:
        save_transaction(
            transaction_type, 
            amount, 
            oldbalanceOrg, 
            newbalanceOrig, 
            oldbalanceDest, 
            newbalanceDest, 
            pred_value
        )
        st.toast("Transaction successfully saved to database!")
    except Exception as e:
        st.error(f"Error saving to database: {e}")