# Streamlit front-end for the Tourism Package Prediction model.
# Loads the trained pipeline from the Hugging Face Model Hub at startup,
# collects customer details via a form, and predicts purchase likelihood.

import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# set_page_config must be the FIRST Streamlit command on the page -- before any
# other st.* call, including the cache spinner emitted by load_model() below.
st.set_page_config(page_title="Tourism Package Prediction", layout="centered")

MODEL_REPO = "creativitysupreme/tourism-predictor"
MODEL_FILE = "best_tourism_model.joblib"


@st.cache_resource(show_spinner=False)   # no spinner -> avoids a startup race that
                                          # can raise "SessionInfo before initialized"
def load_model():
    # Cached so the model is downloaded once per session, not on every click.
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return joblib.load(path)


model = load_model()

# ----------------------------------------------------------------------
# Page header
# ----------------------------------------------------------------------
st.title("Wellness Tourism Package - Purchase Likelihood")
st.write(
    "This internal tool predicts whether a customer is likely to purchase the new "
    "Wellness Tourism Package. Enter the customer details below and click **Predict**."
)

# Threshold control in the sidebar - lets the analyst trade precision for recall.
threshold = st.sidebar.slider(
    "Classification threshold",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Probability above which a customer is flagged as a likely buyer.",
)

# ----------------------------------------------------------------------
# Input form (two columns for compactness)
# ----------------------------------------------------------------------
st.subheader("Customer Details")
col1, col2 = st.columns(2)

with col1:
    Age           = st.number_input("Age", min_value=18, max_value=100, value=35)
    Gender        = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Occupation    = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    )
    Designation   = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    )
    MonthlyIncome = st.number_input(
        "Monthly Income", min_value=1000, max_value=200000, value=25000, step=1000
    )
    CityTier      = st.selectbox("City Tier", [1, 2, 3])
    Passport      = st.selectbox("Holds Passport?", ["No", "Yes"])
    OwnCar        = st.selectbox("Owns Car?", ["No", "Yes"])

with col2:
    NumberOfPersonVisiting   = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=10, value=3
    )
    NumberOfChildrenVisiting = st.number_input(
        "Children Below 5 Visiting", min_value=0, max_value=5, value=0
    )
    PreferredPropertyStar    = st.selectbox("Preferred Hotel Rating", [3, 4, 5])
    NumberOfTrips            = st.number_input(
        "Average Trips per Year", min_value=0, max_value=30, value=3
    )
    TypeofContact            = st.selectbox(
        "Type of Contact", ["Self Enquiry", "Company Invited"]
    )
    ProductPitched           = st.selectbox(
        "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    DurationOfPitch          = st.number_input(
        "Duration of Pitch (min)", min_value=1, max_value=180, value=15
    )
    NumberOfFollowups        = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=10, value=4
    )
    PitchSatisfactionScore   = st.selectbox(
        "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
    )

# ----------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------
if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([{
        "Age": Age,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "PreferredPropertyStar": PreferredPropertyStar,
        "NumberOfTrips": NumberOfTrips,
        "Passport": 1 if Passport == "Yes" else 0,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": 1 if OwnCar == "Yes" else 0,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "MonthlyIncome": MonthlyIncome,
        "TypeofContact": TypeofContact,
        "Occupation": Occupation,
        "Gender": Gender,
        "ProductPitched": ProductPitched,
        "MaritalStatus": MaritalStatus,
        "Designation": Designation,
    }])

    proba = float(model.predict_proba(input_df)[0, 1])
    will_buy = proba >= threshold

    st.subheader("Prediction")
    if will_buy:
        st.success(f"Likely to purchase  ·  probability = {proba:.2%}")
    else:
        st.info(f"Unlikely to purchase  ·  probability = {proba:.2%}")
    st.caption(f"Classification threshold used: {threshold:.2f}")
