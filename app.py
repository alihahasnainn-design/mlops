import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="German Credit Predictor", layout="centered")
st.title("German  Credit Risk Predictor demo")
st.markdown("### Predict if a customer is **Good** or **Bad** credit risk")

@st.cache_resource
def load_model():
    try:
        with open("best_model.pkl", "rb") as f:
            obj = pickle.load(f)
        # Safely extract the sklearn Pipeline object and saved accuracy
        return obj["model"], obj.get("accuracy")
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None, None

model, accuracy = load_model()

if accuracy:
    st.info(f"Model CV Accuracy: **{accuracy*100:.2f}%**")

st.sidebar.header("Enter Customer Details")

def user_input_features():
    age              = st.sidebar.slider("Age (years)", 18, 75, 30)
    sex              = st.sidebar.selectbox("Sex & Marital Status", ["Male (Single)", "Female / Married"])
    job              = st.sidebar.selectbox("Occupation", [0, 1, 2, 3],
                                       format_func=lambda x: ["Unemployed","Unskilled","Skilled","Highly Skilled"][x])
    housing          = st.sidebar.selectbox("Type of apartment", ["Own", "Rent", "Free"])
    saving_account   = st.sidebar.selectbox("Value Savings/Stocks", ["Little", "Moderate", "Quite Rich", "Rich"])
    checking_account = st.sidebar.selectbox("Account Balance", ["Little", "Moderate", "Rich"])
    credit_amount    = st.sidebar.number_input("Credit Amount", min_value=100, max_value=20000, value=1000)
    duration         = st.sidebar.slider("Duration of Credit (month)", 1, 72, 12)
    purpose          = st.sidebar.selectbox("Purpose", [0,1,2,3,4,5,6,7,8,9,10],
                                       format_func=lambda x: ["Car (new)","Car (used)","Furniture","TV","Appliances",
                                                              "Repairs","Education","Vacation","Retraining","Business","Other"][x])

    sex_map      = {"Male (Single)": 1, "Female / Married": 2}
    housing_map  = {"Own": 1, "Rent": 2, "Free": 3}
    saving_map   = {"Little": 1, "Moderate": 2, "Quite Rich": 3, "Rich": 4}
    checking_map = {"Little": 1, "Moderate": 2, "Rich": 3}

    # Column keys must match the EXACT string casing from your MySQL training database
    data = {
        'Creditability':                     1,
        'Account Balance':                  checking_map[checking_account],
        'Duration of Credit (month)':        duration,
        'Payment Status of Previous Credit': 1,
        'Purpose':                           purpose,
        'Credit Amount':                     credit_amount,
        'Value Savings/Stocks':              saving_map[saving_account],
        'Length of current employment':      1,
        'Instalment percent':                1,
        'Sex & Marital Status':              sex_map[sex],
        'Guarantors':                        1,
        'Duration in Current address':       1,
        'Most valuable available asset':     1,
        'Age (years)':                       age,
        'Concurrent Credits':                1,
        'Type of apartment':                 housing_map[housing],
        'No of Credits at this Bank':        1,
        'Occupation':                        job,
        'No of dependents':                  1,
        'Telephone':                         1,
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("Customer Profile Summary")
st.dataframe(input_df, use_container_width=True)

if st.button("🔍 Predict Credit Risk", type="primary"):
    if model is not None:
        with st.spinner("Analyzing profile through ML Pipeline..."):
            try:
                # The model variable here is actually the full scikit-learn Pipeline
                # It will automatically preprocess input_df using ColumnTransformer
                prediction  = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]

                # Adjust mappings depending on how your SQL dataset targets are represented
                if prediction == 1:
                    st.success("✅ Prediction: **GOOD Credit Risk**")
                    st.balloons()
                else:
                    st.error("❌ Prediction: **BAD Credit Risk**")
                    st.warning("⚠️ High Risk: Financial profile suggests potential default risk.")

                col1, col2 = st.columns(2)
                col1.metric("Probability of Good Risk", f"{probability[1]*100:.1f}%")
                col2.metric("Probability of Bad Risk",  f"{probability[0]*100:.1f}%")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.write("**Debug — input shape:**", input_df.shape)
                st.write("**Required Pipeline features:**", input_df.columns.tolist())
    else:
        st.error("Model structure is missing. Deploy 'best_model.pkl' to root.")

st.divider()
st.caption("German Credit Risk Model | Powered by Streamlit")