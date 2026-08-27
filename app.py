import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Configure page settings
st.set_page_config(
    page_title="ICU Mortality Risk Predictor", page_icon="🏥", layout="centered"
)


# ---------------------------------------------------------------------------
# Load model artifacts (cached so they only load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_icu_model.pkl")
    num_imputer = joblib.load("num_imputer.pkl")
    cat_imputer = joblib.load("cat_imputer.pkl")
    with open("final_40_features.json") as f:
        final_40_features = json.load(f)
    with open("chosen_threshold.json") as f:
        threshold = json.load(f)["threshold"]
    with open("encoded_columns.json") as f:
        encoded_columns = json.load(f)
    return (
        model,
        num_imputer,
        cat_imputer,
        final_40_features,
        threshold,
        encoded_columns,
    )


model, num_imputer, cat_imputer, final_40_features, threshold, encoded_columns = (
    load_artifacts()
)

num_cols = list(num_imputer.feature_names_in_)
num_defaults = dict(zip(num_cols, num_imputer.statistics_))

# ---------------------------------------------------------------------------
# Initialize Session State (Memory for the auto-fill)
# ---------------------------------------------------------------------------
for col in num_cols:
    if col not in st.session_state:
        st.session_state[col] = float(num_defaults[col])

if "ventilated_status" not in st.session_state:
    st.session_state["ventilated_status"] = "No"

# ---------------------------------------------------------------------------
# UI Header & CSV Auto-Fill Uploader
# ---------------------------------------------------------------------------
st.title("🏥 ICU Mortality Risk Predictor")
st.caption(
    "Estimates in-hospital mortality risk from ICU admission data. "
    "Fields left blank default to typical (median) baseline values from the training data."
)
st.divider()

st.markdown("### 📥 Auto-Fill Form from CSV")
st.info(
    "Upload a patient CSV. The form below will automatically fill with the data from the first row so you don't have to type it manually."
)
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if not df.empty:
            row_data = df.iloc[0]  # Grab the first patient

            # Update session state with CSV values
            for col in df.columns:
                if col in st.session_state:
                    st.session_state[col] = float(row_data[col])

            if "ventilated_apache" in df.columns:
                st.session_state["ventilated_status"] = (
                    "Yes" if int(row_data["ventilated_apache"]) == 1 else "No"
                )

            st.success(
                "✅ Form successfully auto-filled! You can now review or edit the inputs below before predicting."
            )
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

st.write("")  # Spacing

# ---------------------------------------------------------------------------
# Form & Layout (Pre-filled by Session State)
# ---------------------------------------------------------------------------
with st.form("patient_form"):
    # --- SECTION 1: Clinical Scores ---
    st.markdown("### 📋 1. Clinical Scores & GCS")
    c1, c2 = st.columns(2)
    with c1:
        apache_4a_icu_death_prob = st.number_input(
            "APACHE IVa ICU death probability",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["apache_4a_icu_death_prob"]),
            step=0.01,
        )
        apache_2_diagnosis = st.number_input(
            "APACHE II diagnosis code",
            value=float(st.session_state["apache_2_diagnosis"]),
            step=1.0,
        )
        gcs_motor_apache = st.slider(
            "Motor response", 1, 6, int(st.session_state["gcs_motor_apache"])
        )
        gcs_verbal_apache = st.slider(
            "Verbal response", 1, 5, int(st.session_state["gcs_verbal_apache"])
        )
    with c2:
        apache_4a_hospital_death_prob = st.number_input(
            "APACHE IVa hospital death probability",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["apache_4a_hospital_death_prob"]),
            step=0.01,
        )
        apache_3j_diagnosis = st.number_input(
            "APACHE IIIj diagnosis code",
            value=float(st.session_state["apache_3j_diagnosis"]),
            step=1.0,
        )
        gcs_eyes_apache = st.slider(
            "Eye opening", 1, 4, int(st.session_state["gcs_eyes_apache"])
        )

        vent_options = ["No", "Yes"]
        vent_index = vent_options.index(st.session_state["ventilated_status"])
        ventilated_apache = st.selectbox(
            "Ventilated on admission?", vent_options, index=vent_index
        )

    st.write("")

    # --- SECTION 2: Vitals ---
    st.markdown("### 🫀 2. Vitals — First Day (Day 1) Min/Max")
    c1, c2 = st.columns(2)
    with c1:
        d1_heartrate_max = st.number_input(
            "Heart rate — max (bpm)", value=float(st.session_state["d1_heartrate_max"])
        )
        d1_sysbp_min = st.number_input(
            "Systolic BP — min (mmHg)", value=float(st.session_state["d1_sysbp_min"])
        )
        d1_sysbp_noninvasive_min = st.number_input(
            "Systolic BP (non-inv) — min",
            value=float(st.session_state["d1_sysbp_noninvasive_min"]),
        )
        d1_mbp_min = st.number_input(
            "Mean BP — min (mmHg)", value=float(st.session_state["d1_mbp_min"])
        )
        d1_mbp_noninvasive_min = st.number_input(
            "Mean BP (non-inv) — min",
            value=float(st.session_state["d1_mbp_noninvasive_min"]),
        )
        d1_diasbp_min = st.number_input(
            "Diastolic BP — min (mmHg)", value=float(st.session_state["d1_diasbp_min"])
        )
        d1_diasbp_noninvasive_min = st.number_input(
            "Diastolic BP (non-inv) — min",
            value=float(st.session_state["d1_diasbp_noninvasive_min"]),
        )
    with c2:
        d1_heartrate_min = st.number_input(
            "Heart rate — min (bpm)", value=float(st.session_state["d1_heartrate_min"])
        )
        d1_spo2_min = st.number_input(
            "SpO2 — min (%)", value=float(st.session_state["d1_spo2_min"])
        )
        d1_spo2_max = st.number_input(
            "SpO2 — max (%)", value=float(st.session_state["d1_spo2_max"])
        )
        d1_temp_min = st.number_input(
            "Temperature — min (°C)", value=float(st.session_state["d1_temp_min"])
        )
        d1_temp_max = st.number_input(
            "Temperature — max (°C)", value=float(st.session_state["d1_temp_max"])
        )
        h1_resprate_min = st.number_input(
            "Resp. rate (hour 1) — min",
            value=float(st.session_state["h1_resprate_min"]),
        )
        d1_resprate_min = st.number_input(
            "Resp. rate (day 1) — min", value=float(st.session_state["d1_resprate_min"])
        )
        h1_resprate_max = st.number_input(
            "Resp. rate (hour 1) — max",
            value=float(st.session_state["h1_resprate_max"]),
        )

    st.write("")

    # --- SECTION 3: Labs ---
    st.markdown("### 🧪 3. Laboratory Results")
    c1, c2 = st.columns(2)
    with c1:
        bun_apache = st.number_input(
            "BUN (APACHE)", value=float(st.session_state["bun_apache"])
        )
        d1_bun_max = st.number_input(
            "BUN — max", value=float(st.session_state["d1_bun_max"])
        )
        d1_bun_min = st.number_input(
            "BUN — min", value=float(st.session_state["d1_bun_min"])
        )
        creatinine_apache = st.number_input(
            "Creatinine (APACHE)", value=float(st.session_state["creatinine_apache"])
        )
        d1_creatinine_max = st.number_input(
            "Creatinine — max", value=float(st.session_state["d1_creatinine_max"])
        )
        d1_creatinine_min = st.number_input(
            "Creatinine — min", value=float(st.session_state["d1_creatinine_min"])
        )
    with c2:
        temp_apache = st.number_input(
            "Temperature (APACHE)", value=float(st.session_state["temp_apache"])
        )
        d1_hco3_min = st.number_input(
            "HCO3 — min", value=float(st.session_state["d1_hco3_min"])
        )
        d1_hco3_max = st.number_input(
            "HCO3 — max", value=float(st.session_state["d1_hco3_max"])
        )
        d1_wbc_max = st.number_input(
            "WBC — max", value=float(st.session_state["d1_wbc_max"])
        )
        d1_platelets_min = st.number_input(
            "Platelets — min", value=float(st.session_state["d1_platelets_min"])
        )

    st.write("")
    submitted = st.form_submit_button(
        "Predict Mortality Risk", type="primary", use_container_width=True
    )

# ---------------------------------------------------------------------------
# Prediction & Visual Dashboard
# ---------------------------------------------------------------------------
if submitted:
    # Build dictionary starting from defaults, overriding with user input
    row = dict(num_defaults)
    row.update(
        {
            "apache_4a_icu_death_prob": apache_4a_icu_death_prob,
            "apache_4a_hospital_death_prob": apache_4a_hospital_death_prob,
            "apache_2_diagnosis": apache_2_diagnosis,
            "apache_3j_diagnosis": apache_3j_diagnosis,
            "gcs_motor_apache": gcs_motor_apache,
            "gcs_eyes_apache": gcs_eyes_apache,
            "gcs_verbal_apache": gcs_verbal_apache,
            "ventilated_apache": 1 if ventilated_apache == "Yes" else 0,
            "d1_heartrate_max": d1_heartrate_max,
            "d1_heartrate_min": d1_heartrate_min,
            "d1_sysbp_min": d1_sysbp_min,
            "d1_sysbp_noninvasive_min": d1_sysbp_noninvasive_min,
            "d1_mbp_min": d1_mbp_min,
            "d1_mbp_noninvasive_min": d1_mbp_noninvasive_min,
            "d1_diasbp_min": d1_diasbp_min,
            "d1_diasbp_noninvasive_min": d1_diasbp_noninvasive_min,
            "d1_spo2_min": d1_spo2_min,
            "d1_spo2_max": d1_spo2_max,
            "d1_temp_min": d1_temp_min,
            "d1_temp_max": d1_temp_max,
            "h1_resprate_min": h1_resprate_min,
            "d1_resprate_min": d1_resprate_min,
            "h1_resprate_max": h1_resprate_max,
            "bun_apache": bun_apache,
            "d1_bun_max": d1_bun_max,
            "d1_bun_min": d1_bun_min,
            "creatinine_apache": creatinine_apache,
            "d1_creatinine_max": d1_creatinine_max,
            "d1_creatinine_min": d1_creatinine_min,
            "temp_apache": temp_apache,
            "d1_hco3_min": d1_hco3_min,
            "d1_hco3_max": d1_hco3_max,
            "d1_wbc_max": d1_wbc_max,
            "d1_platelets_min": d1_platelets_min,
        }
    )

    # DataFrame and Feature Engineering
    X_row = pd.DataFrame([row])[num_cols]
    X_row["shock_index"] = X_row["d1_heartrate_max"] / (X_row["d1_sysbp_min"] + 1e-5)
    X_row["spo2_resprate_ratio"] = X_row["d1_spo2_min"] / (
        X_row["h1_resprate_max"] + 1e-5
    )
    X_row["gcs_total_score"] = (
        X_row["gcs_motor_apache"]
        + X_row["gcs_eyes_apache"]
        + X_row["gcs_verbal_apache"]
    )
    X_row["d1_heartrate_range"] = X_row["d1_heartrate_max"] - X_row["d1_heartrate_min"]
    X_row["d1_temp_range"] = X_row["d1_temp_max"] - X_row["d1_temp_min"]
    X_row["d1_spo2_range"] = X_row["d1_spo2_max"] - X_row["d1_spo2_min"]
    X_row["d1_bun_max_log"] = np.log1p(X_row["d1_bun_max"])
    X_row["d1_creatinine_max_log"] = np.log1p(X_row["d1_creatinine_max"])

    X_final = X_row[final_40_features].fillna(0)

    # Predictions
    proba = model.predict_proba(X_final)[:, 1][0]
    prediction = int(proba >= threshold)

    # --- UI: Results & Plots ---
    st.divider()
    st.markdown("## 📊 Prediction Dashboard")

    if prediction == 1:
        st.error(
            f"⚠️ **High Risk Flag** — Model predicts elevated mortality risk. (Probability: {proba:.1%})"
        )
    else:
        st.success(
            f"✅ **Lower Risk** — Model does not flag elevated mortality risk. (Probability: {proba:.1%})"
        )

    c1, c2 = st.columns([1, 1])

    with c1:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=proba * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                delta={
                    "reference": threshold * 100,
                    "increasing": {"color": "red"},
                    "decreasing": {"color": "green"},
                },
                title={"text": "Mortality Risk vs Threshold", "font": {"size": 18}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "darkblue",
                    },
                    "bar": {"color": "rgba(0,0,0,0.5)", "thickness": 0.25},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {
                            "range": [0, threshold * 100],
                            "color": "rgba(144, 238, 144, 0.4)",
                        },
                        {
                            "range": [threshold * 100, 100],
                            "color": "rgba(250, 128, 114, 0.4)",
                        },
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": threshold * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": final_40_features, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False).head(8)

        fig_bar = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Model's Top Driving Factors",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=320,
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.caption(
        "**Disclaimer:** This tool provides a decision-support estimate based on historical machine learning data, not a medical diagnosis. "
        "Always use clinical judgment alongside this output."
    )
