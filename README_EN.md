
# Planet Classification Project

## Overview
This project aims to **classify planets** based on their characteristics into two categories:  

- **Class 0:** Most common type of planets  
- **Class 1:** Rare or special type of planets  

The dataset contains features like planetary mass, radius, orbital parameters, and host star characteristics. Due to the **highly imbalanced nature** of the data (minor class ~5%), special techniques were applied to improve the detection of rare planets.

---

## Dataset
- Columns include:
  - `PlanetaryMassJpt` – Mass of the planet in Jupiter masses  
  - `RadiusJpt` – Radius of the planet in Jupiter radii  
  - `PeriodDays` – Orbital period in days  
  - `SemiMajorAxisAU` – Semi-major axis in AU  
  - `Eccentricity` – Orbital eccentricity  
  - `HostStarMassSlrMass` – Mass of the host star in solar masses  
  - `HostStarRadiusSlrRad` – Radius of the host star in solar radii  
  - `TypeFlag` – Original categorical class  

- **Target variable:** `TargetBinary`  
  - 0 → original `TypeFlag = 0`  
  - 1 → original `TypeFlag != 0`  

- **Imbalance:** ~2717 major class vs 150 minor class (before SMOTE)

---

## Feature Engineering
Several derived features were created to improve model performance:  

- `EccentricitySquared` – square of eccentricity  
- `OrbitalEnergy` – approximate orbital energy  
- `SemiMajorAxisLog` – log transformation of semi-major axis  
- `ScaledPeriod` – normalized period  
- `RadiusSqrt` / `MassSqrt` – square root transformations  
- `MassRadiusRatio`, `DensityApprox` – derived from mass and radius  
- `PeriodLog` – log-transformed period  

These features were essential to **highlight patterns** of the minor class planets.

---

## Methodology
1. **Data Preprocessing**  
   - Handle missing values  
   - Split dataset into training (80%) and test (20%)  

2. **Imbalanced Data Handling**  
   - Applied **SMOTE** to oversample the minor class in training set  

3. **Model**  
   - **CatBoostClassifier**  
   - Parameters:
     - `iterations=500`  
     - `depth=5`  
     - `learning_rate=0.05`  
     - `subsample=0.8`  
     - `colsample_bylevel=0.8`  

4. **Threshold Tuning**  
   - Probability threshold for predicting minor class optimized for **F1 score of class 1**  

5. **Evaluation Metrics**  
   - Precision, Recall, F1-score  
   - Macro F1  
   - Balanced Accuracy  
   - Confusion Matrix  

---

## Results
- **Best threshold (minor class):** 0.33  
- **Minor class F1 score:** 0.34  
- **Minor class recall:** 0.66  
- **Major class precision/recall:** 0.98 / 0.87  
- **Balanced Accuracy:** 0.77  

- **Top features (importance):**  
  1. `EccentricitySquared`  
  2. `Eccentricity`  
  3. `OrbitalEnergy`  
  4. `SemiMajorAxisAU`  
  5. `HostStarMassSlrMass`  

> The model demonstrates **effective detection of rare planets** despite high imbalance, with interpretable feature importance.

---

## Conclusion
This project showcases:  

- Handling **imbalanced datasets** using SMOTE  
- Importance of **feature engineering** to boost minor class detection  
- **Threshold tuning** to optimize minor class recall  
- Use of **CatBoostClassifier** for tabular data with good interpretability  

This workflow can be adapted for other **imbalanced classification problems** in astronomy or general tabular datasets.

---

## Next Steps / Optional Improvements
- Visualize **confusion matrix** and **feature importance**  
- Try **focal loss** or other boosting methods to further improve minor class precision  
- Cross-validation for more **robust evaluation**  