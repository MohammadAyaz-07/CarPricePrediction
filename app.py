
import streamlit as st
import pandas as pd

from pathlib import Path
from datetime import date

from src.prediction import predict_car_price, VALID_CATEGORIES


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "used_cars_cleaned.csv"


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AutoVal | Vehicle Valuation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# CUSTOM DESIGN SYSTEM
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F6F5F1;
    color: #20252B;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background-color: transparent;
}

#MainMenu,
footer {
    visibility: hidden;
}


/* BRANDING */

.brand {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #20252B;
}

.brand span {
    color: #C58B38;
}


/* HERO */

.eyebrow {
    color: #A8752D;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -2px;
    color: #20252B;
    margin-top: 12px;
}

.hero-description {
    font-size: 16px;
    color: #707780;
    line-height: 1.7;
    max-width: 680px;
}


/* SECTION HEADINGS */

.section-title {
    font-size: 23px;
    font-weight: 750;
    color: #20252B;
    letter-spacing: -0.5px;
}

.section-description {
    font-size: 14px;
    color: #777F87;
}


/* FORM CONTAINER */

div[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #E8E5DF;
    border-radius: 15px;
    padding: 25px;
}

div[data-testid="stForm"] label {
    color: #333A42 !important;
    font-weight: 600;
}


/* DROPDOWN STYLING */

div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {
    background-color: #FFFFFF !important;
    border: 1px solid #DAD7D0 !important;
    border-radius: 8px !important;
}

div[data-testid="stSelectbox"] [role="combobox"] {
    background-color: #FFFFFF !important;
    color: #20252B !important;
}

div[data-testid="stSelectbox"] [role="combobox"] * {
    color: #20252B !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #555D65 !important;
}


/* DROPDOWN MENU OPTIONS */

div[data-baseweb="popover"] ul {
    background-color: #FFFFFF !important;
}

div[data-baseweb="popover"] li {
    background-color: #FFFFFF !important;
    color: #20252B !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #F6F5F1 !important;
}


/* NUMBER INPUTS */

div[data-testid="stNumberInput"] [data-baseweb="input"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #DAD7D0 !important;
    border-radius: 8px !important;
}

div[data-testid="stNumberInput"] input {
    color: #20252B !important;
    background-color: #FFFFFF !important;
}

div[data-testid="stNumberInput"] button {
    color: #20252B !important;
    background-color: #F6F5F1 !important;
}


/* INPUT FOCUS */

div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
div[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
    border-color: #C58B38 !important;
    box-shadow: 0 0 0 1px #C58B38 !important;
}


/* PREDICTION BUTTON */

div[data-testid="stFormSubmitButton"] button {
    background-color: #252B32 !important;
    color: #FFFFFF !important;
    border: 1px solid #252B32 !important;
    border-radius: 8px !important;
    min-height: 52px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    transition: background-color 0.2s ease;
}

div[data-testid="stFormSubmitButton"] button p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #C58B38 !important;
    border-color: #C58B38 !important;
}


/* RESULT CARD */

.result-card {
    background-color: #20252B;
    border-radius: 15px;
    padding: 30px;
    color: #FFFFFF;
    margin-top: 25px;
}

.result-label {
    color: #D6B477;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.result-price {
    color: #FFFFFF;
    font-size: 43px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-top: 10px;
}

.result-note {
    color: #C3C8CD;
    font-size: 13px;
}


/* FOOTER */

.footer-text {
    color: #92989E;
    font-size: 12px;
    text-align: center;
}

div[data-testid="stDivider"] {
    border-color: #E5E2DC;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD REFERENCE DATA
# ==========================================

@st.cache_data
def load_reference_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


try:
    df = load_reference_data()

except Exception as error:
    st.error(f"Unable to load vehicle dataset: {error}")
    st.stop()


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_default_category(feature, preferred):

    options = sorted(VALID_CATEGORIES[feature])

    if preferred in options:
        return options.index(preferred)

    return 0


def get_numeric_default(feature):

    return float(df[feature].median())


def get_numeric_max(feature):

    return max(float(df[feature].max()) * 1.5, 1.0)


# ==========================================
# NAVIGATION HEADER
# ==========================================

header_left, header_right = st.columns([3, 1])

with header_left:

    st.markdown(
        '<div class="brand">AUTO<span>VAL.</span></div>',
        unsafe_allow_html=True
    )

with header_right:

    st.markdown(
        '<p style="text-align:right; color:#777F87; '
        'font-size:12px; font-weight:700; '
        'letter-spacing:1px;">VEHICLE VALUATION</p>',
        unsafe_allow_html=True
    )

st.divider()


# ==========================================
# HERO SECTION
# ==========================================

st.markdown(
    '<div class="eyebrow">INTELLIGENT VEHICLE VALUATION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">'
    'Know what your car<br>is really worth.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="hero-description">'
    'Get an estimated resale value for your used car '
    'using vehicle specifications and a machine learning model '
    'trained on Indian used-car data.'
    '</p>',
    unsafe_allow_html=True
)

st.write("")


# ==========================================
# VEHICLE INFORMATION FORM
# ==========================================

st.markdown(
    '<div class="section-title">Vehicle information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Provide the details below to calculate an estimated resale value.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


with st.form("vehicle_prediction_form"):

    # BASIC INFORMATION

    st.markdown("#### 01  ·  Basic information")

    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox(
            "Vehicle brand",
            options=sorted(VALID_CATEGORIES["brand"]),
            index=get_default_category("brand", "Maruti")
        )

    with col2:
        year = st.number_input(
            "Manufacturing year",
            min_value=int(df["year"].min()),
            max_value=date.today().year,
            value=int(round(get_numeric_default("year"))),
            step=1
        )

    with col3:
        owner = st.selectbox(
            "Ownership history",
            options=sorted(VALID_CATEGORIES["owner"]),
            index=get_default_category("owner", "First Owner")
        )

    st.divider()


    # ENGINE AND PERFORMANCE

    st.markdown("#### 02  ·  Engine and performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        fuel = st.selectbox(
            "Fuel type",
            options=sorted(VALID_CATEGORIES["fuel"]),
            index=get_default_category("fuel", "Petrol")
        )

    with col2:
        engine = st.number_input(
            "Engine capacity (CC)",
            min_value=0.0,
            max_value=get_numeric_max("engine"),
            value=get_numeric_default("engine"),
            step=50.0
        )

    with col3:
        max_power = st.number_input(
            "Maximum power (BHP)",
            min_value=0.0,
            max_value=get_numeric_max("max_power"),
            value=get_numeric_default("max_power"),
            step=1.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        mileage = st.number_input(
            "Mileage (km/l or km/kg)",
            min_value=0.0,
            max_value=get_numeric_max("mileage"),
            value=get_numeric_default("mileage"),
            step=0.5
        )

    with col2:
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=get_numeric_max("torque"),
            value=get_numeric_default("torque"),
            step=5.0
        )

    with col3:
        seats = st.number_input(
            "Seating capacity",
            min_value=1,
            max_value=max(int(df["seats"].max()), 1),
            value=int(round(get_numeric_default("seats"))),
            step=1
        )

    st.divider()


    # USAGE AND CONFIGURATION

    st.markdown("#### 03  ·  Usage and configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        km_driven = st.number_input(
            "Kilometres driven",
            min_value=0,
            max_value=max(int(df["km_driven"].max() * 1.5), 1),
            value=int(round(get_numeric_default("km_driven"))),
            step=1000
        )

    with col2:
        transmission = st.selectbox(
            "Transmission",
            options=sorted(VALID_CATEGORIES["transmission"]),
            index=get_default_category("transmission", "Manual")
        )

    with col3:
        seller_type = st.selectbox(
            "Seller type",
            options=sorted(VALID_CATEGORIES["seller_type"]),
            index=get_default_category("seller_type", "Individual")
        )

    st.write("")

    submitted = st.form_submit_button(
        "ESTIMATE MY CAR PRICE  →",
        use_container_width=True
    )


# ==========================================
# PREDICTION RESULT
# ==========================================

if submitted:

    vehicle_data = {
        "year": year,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "torque": torque,
        "seats": seats,
        "brand": brand,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner
    }

    try:

        predicted_price = predict_car_price(vehicle_data)

        st.write("")

        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-label">'
            f'YOUR ESTIMATED RESALE VALUE'
            f'</div>'
            f'<div class="result-price">'
            f'₹{predicted_price:,.0f}'
            f'</div>'
            f'<div class="result-note">'
            f'Estimated price based on the supplied vehicle specifications.'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "This is a machine learning estimate, not a guaranteed "
            "market offer. Actual resale prices may vary."
        )

    except (ValueError, TypeError) as error:

        st.error(
            f"Unable to generate prediction: {error}"
        )

    except Exception as error:

        st.error(
            "An unexpected error occurred. "
            "Please check the application configuration."
        )

        st.exception(error)


# ==========================================
# FOOTER
# ==========================================

st.write("")
st.divider()

st.markdown(
    '<div class="footer-text">'
    'AUTOVAL · INTELLIGENT VEHICLE VALUATION<br>'
    'Machine learning powered used-car price estimation'
    '</div>',
    unsafe_allow_html=True
)
