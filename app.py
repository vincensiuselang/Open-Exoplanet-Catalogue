import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Load model & threshold
# ---------------------------
model = pickle.load(open("models/catboost_model.pkl", "rb"))
best_threshold = pickle.load(open("models/best_threshold.pkl", "rb"))

st.set_page_config(page_title="Planet Class Predictor", layout="wide")
st.title("🌌 Planet Class Predictor")

# ---------------------------
# 2️⃣ Sidebar input
# ---------------------------
st.sidebar.header("Input Planet Data")

PlanetaryMassJpt = st.sidebar.number_input("Planetary Mass (Jupiter Masses)", value=0.5)
RadiusJpt = st.sidebar.number_input("Planet Radius (Jupiter Radii)", value=0.9)
PeriodDays = st.sidebar.number_input("Orbital Period (Days)", value=50)
SemiMajorAxisAU = st.sidebar.number_input("Semi-Major Axis (AU)", value=0.3)
Eccentricity = st.sidebar.number_input("Eccentricity", value=0.05)
HostStarMassSlrMass = st.sidebar.number_input("Host Star Mass (Solar Mass)", value=1.0)
HostStarRadiusSlrRad = st.sidebar.number_input("Host Star Radius (Solar Radii)", value=1.0)

# ---------------------------
# 3️⃣ Feature engineering
# ---------------------------
X_new = pd.DataFrame([{
    'PlanetaryMassJpt': PlanetaryMassJpt,
    'RadiusJpt': RadiusJpt,
    'PeriodDays': PeriodDays,
    'SemiMajorAxisAU': SemiMajorAxisAU,
    'Eccentricity': Eccentricity,
    'HostStarMassSlrMass': HostStarMassSlrMass,
    'HostStarRadiusSlrRad': HostStarRadiusSlrRad,
    'EccentricitySquared': Eccentricity**2,
    'OrbitalEnergy': HostStarMassSlrMass / SemiMajorAxisAU,
    'SemiMajorAxisLog': np.log1p(SemiMajorAxisAU),
    'ScaledPeriod': PeriodDays / 365,
    'RadiusSqrt': np.sqrt(RadiusJpt),
    'MassSqrt': np.sqrt(PlanetaryMassJpt),
    'MassRadiusRatio': PlanetaryMassJpt / RadiusJpt,
    'DensityApprox': PlanetaryMassJpt / RadiusJpt**3,
    'PeriodLog': np.log1p(PeriodDays)
}])

features = [
    'EccentricitySquared', 'Eccentricity', 'OrbitalEnergy', 'SemiMajorAxisAU',
    'HostStarMassSlrMass', 'SemiMajorAxisLog', 'HostStarRadiusSlrRad',
    'ScaledPeriod', 'PlanetaryMassJpt', 'DensityApprox', 'RadiusJpt',
    'MassSqrt', 'RadiusSqrt', 'PeriodLog', 'MassRadiusRatio', 'PeriodDays'
]

X_new = X_new[features]

# ---------------------------
# 4️⃣ Prediction
# ---------------------------
y_proba = model.predict_proba(X_new)[:, 1]
y_pred = (y_proba > best_threshold).astype(int)

st.subheader("Prediction Result")
st.write(f"**Predicted Class:** {'Minor/Rare' if y_pred[0]==1 else 'Major/Common'}")
st.write(f"**Probability (Minor Class):** {y_proba[0]:.2f}")

# ---------------------------
# 5️⃣ Probability gauge (bar)
# ---------------------------
st.subheader("Probability Gauge")
fig, ax = plt.subplots(figsize=(6,1))
ax.barh([0], [y_proba[0]], color='skyblue')
ax.set_xlim(0,1)
ax.set_yticks([])
ax.set_xlabel("Probability (Minor Class)")
ax.axvline(best_threshold, color='red', linestyle='--', label=f'Threshold = {best_threshold:.2f}')
ax.legend()
st.pyplot(fig)

# ---------------------------
# 6️⃣ Feature importance (top 10)
# ---------------------------
st.subheader("Top 10 Feature Importances")
importance = model.feature_importances_
fi = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values('importance', ascending=False).head(10)

fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.barh(fi['feature'], fi['importance'], color='lightgreen')
ax2.invert_yaxis()
ax2.set_xlabel("Importance")
st.pyplot(fig2)
