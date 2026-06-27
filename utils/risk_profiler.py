"""
utils/risk_profiler.py
Epidemiological breast cancer risk scoring using the Gail Model (simplified)
+ lifestyle factors. Completely separate from the scan ML model.
"""

import numpy as np


# ─── Gail Model Base Risk Table (simplified annual incidences by age) ────────
# Source: NCI BCRAT model approximation
GAIL_BASE = {
    (20, 24): 0.0006, (25, 29): 0.0009, (30, 34): 0.0015,
    (35, 39): 0.0024, (40, 44): 0.0038, (45, 49): 0.0054,
    (50, 54): 0.0068, (55, 59): 0.0080, (60, 64): 0.0089,
    (65, 69): 0.0097, (70, 74): 0.0102, (75, 79): 0.0105,
    (80, 84): 0.0104,
}


def get_base_rate(age: int) -> float:
    for (lo, hi), rate in GAIL_BASE.items():
        if lo <= age <= hi:
            return rate
    return 0.01


# ─── Risk Factor Multipliers ─────────────────────────────────────────────────

MENARCHE_RR = {
    "≥ 14 years": 1.00,
    "12–13 years": 1.10,
    "< 12 years":  1.21,
}

FIRST_BIRTH_RR = {
    "No children": 1.20,
    "< 20 years":  1.00,
    "20–24 years": 1.11,
    "25–29 years": 1.24,
    "≥ 30 years":  1.55,
}

RELATIVES_RR = {
    0: 1.00,
    1: 1.80,
    2: 2.90,
}

BIOPSY_RR = {
    "Never":                           1.00,
    "Yes – no atypical hyperplasia":   1.70,
    "Yes – atypical hyperplasia":      4.24,
}

DENSITY_RR = {
    "Fatty (low density)":             0.80,
    "Scattered fibroglandular":        1.00,
    "Heterogeneously dense":           1.20,
    "Extremely dense":                 2.00,
}

HRT_RR = {
    "Never":                           1.00,
    "Estrogen only (< 5 yrs)":         1.10,
    "Estrogen only (≥ 5 yrs)":         1.30,
    "Combined HRT (< 5 yrs)":          1.20,
    "Combined HRT (≥ 5 yrs)":          1.75,
}

ALCOHOL_RR = {
    "None":                            1.00,
    "< 1 drink/day":                   1.05,
    "1 drink/day":                     1.10,
    "> 1 drink/day":                   1.40,
}

BMI_RR = {
    "Underweight (< 18.5)":            0.90,
    "Normal (18.5–24.9)":              1.00,
    "Overweight (25–29.9)":            1.12,
    "Obese (≥ 30)":                    1.26,
}

EXERCISE_RR = {
    "≥ 4 hrs/week vigorous":           0.80,
    "2–4 hrs/week moderate":           0.90,
    "< 2 hrs/week":                    1.00,
    "Sedentary":                       1.10,
}

BRCA_RR = {
    "Unknown / not tested":            1.00,
    "Negative":                        0.90,
    "BRCA1 positive":                  7.00,
    "BRCA2 positive":                  5.50,
}


# ─── Score Engine ────────────────────────────────────────────────────────────

def compute_risk_score(inputs: dict) -> dict:
    """
    Compute 5-year and 10-year absolute risk estimates.

    inputs keys:
      age, menarche_age, first_birth, num_relatives,
      biopsy, breast_density, hrt, alcohol, bmi_cat,
      exercise, brca
    """
    age = inputs["age"]
    base_annual = get_base_rate(age)

    # Combined relative risk from all factors
    rr = 1.0
    rr *= MENARCHE_RR.get(inputs["menarche_age"],   1.0)
    rr *= FIRST_BIRTH_RR.get(inputs["first_birth"],  1.0)
    rr *= RELATIVES_RR.get(min(inputs["num_relatives"], 2), 1.0)
    rr *= BIOPSY_RR.get(inputs["biopsy"],            1.0)
    rr *= DENSITY_RR.get(inputs["breast_density"],   1.0)
    rr *= HRT_RR.get(inputs["hrt"],                  1.0)
    rr *= ALCOHOL_RR.get(inputs["alcohol"],          1.0)
    rr *= BMI_RR.get(inputs["bmi_cat"],              1.0)
    rr *= EXERCISE_RR.get(inputs["exercise"],        1.0)
    rr *= BRCA_RR.get(inputs["brca"],                1.0)

    adjusted_annual = base_annual * rr

    # Cumulative risk (compound)
    risk_5yr  = (1 - (1 - adjusted_annual) ** 5)  * 100
    risk_10yr = (1 - (1 - adjusted_annual) ** 10) * 100
    risk_life = (1 - (1 - adjusted_annual) ** max(0, 80 - age)) * 100

    # Average woman baseline for comparison
    avg_5yr  = (1 - (1 - base_annual) ** 5)  * 100
    avg_10yr = (1 - (1 - base_annual) ** 10) * 100

    # Risk tier
    if risk_10yr < 5:
        tier, tier_css, advice = "Low", "low", "Your risk is below average. Maintain a healthy lifestyle and attend routine screenings."
    elif risk_10yr < 15:
        tier, tier_css, advice = "Moderate", "moderate", "Moderate risk detected. Discuss screening frequency with your doctor. Lifestyle improvements can help."
    elif risk_10yr < 30:
        tier, tier_css, advice = "High", "high", "High risk profile. Annual mammograms and MRI screening are recommended. Consult an oncologist."
    else:
        tier, tier_css, advice = "Very High", "critical", "Very high risk. Genetic counseling, chemoprevention options, and enhanced screening are strongly advised."

    # Individual factor contributions (% increase over baseline)
    factor_contributions = {
        "Age":                  round((get_base_rate(age) / 0.005 - 1) * 20, 1),
        "Menarche Age":         round((MENARCHE_RR.get(inputs["menarche_age"], 1) - 1) * 100, 1),
        "First Birth":          round((FIRST_BIRTH_RR.get(inputs["first_birth"], 1) - 1) * 100, 1),
        "Family History":       round((RELATIVES_RR.get(min(inputs["num_relatives"], 2), 1) - 1) * 100, 1),
        "Biopsy History":       round((BIOPSY_RR.get(inputs["biopsy"], 1) - 1) * 100, 1),
        "Breast Density":       round((DENSITY_RR.get(inputs["breast_density"], 1) - 1) * 100, 1),
        "HRT Use":              round((HRT_RR.get(inputs["hrt"], 1) - 1) * 100, 1),
        "Alcohol":              round((ALCOHOL_RR.get(inputs["alcohol"], 1) - 1) * 100, 1),
        "BMI":                  round((BMI_RR.get(inputs["bmi_cat"], 1) - 1) * 100, 1),
        "Exercise":             round((EXERCISE_RR.get(inputs["exercise"], 1) - 1) * 100, 1),
        "BRCA Status":          round((BRCA_RR.get(inputs["brca"], 1) - 1) * 100, 1),
    }

    return {
        "risk_5yr":             round(risk_5yr,  2),
        "risk_10yr":            round(risk_10yr, 2),
        "risk_lifetime":        round(min(risk_life, 85), 2),
        "avg_5yr":              round(avg_5yr,  2),
        "avg_10yr":             round(avg_10yr, 2),
        "relative_risk":        round(rr, 2),
        "tier":                 tier,
        "tier_css":             tier_css,
        "advice":               advice,
        "factor_contributions": factor_contributions,
    }


def get_modifiable_factors(inputs: dict) -> list:
    """Return list of modifiable risk factors the user can improve."""
    tips = []
    if inputs["alcohol"] in ["> 1 drink/day", "1 drink/day"]:
        tips.append(("🍷 Alcohol", "Reducing alcohol intake to < 1 drink/day can lower risk by ~10–40%.", "high"))
    if inputs["bmi_cat"] in ["Overweight (25–29.9)", "Obese (≥ 30)"]:
        tips.append(("⚖️ BMI", "Achieving a healthy BMI reduces post-menopausal breast cancer risk by up to 26%.", "medium"))
    if inputs["exercise"] in ["Sedentary", "< 2 hrs/week"]:
        tips.append(("🏃 Exercise", "≥ 4 hours/week of vigorous exercise is associated with a 20% risk reduction.", "high"))
    if inputs["hrt"] in ["Combined HRT (≥ 5 yrs)", "Estrogen only (≥ 5 yrs)"]:
        tips.append(("💊 HRT", "Long-term HRT significantly raises risk. Discuss alternatives with your doctor.", "high"))
    if not tips:
        tips.append(("✅ Lifestyle", "Your modifiable risk factors look good! Keep up the healthy habits.", "low"))
    return tips


# ─── Factor options for UI ────────────────────────────────────────────────────

UI_OPTIONS = {
    "menarche_age":    list(MENARCHE_RR.keys()),
    "first_birth":     list(FIRST_BIRTH_RR.keys()),
    "biopsy":          list(BIOPSY_RR.keys()),
    "breast_density":  list(DENSITY_RR.keys()),
    "hrt":             list(HRT_RR.keys()),
    "alcohol":         list(ALCOHOL_RR.keys()),
    "bmi_cat":         list(BMI_RR.keys()),
    "exercise":        list(EXERCISE_RR.keys()),
    "brca":            list(BRCA_RR.keys()),
}