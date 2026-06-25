# groundwater-depletion-forecasting-western-UP
AI-powered groundwater level depletion forecasting &amp; risk classification for 6 districts in Western Uttar Pradesh | Random Forest + Linear Trend | CGWB 1996-2017 | SDG 6 &amp; SDG 13
# Groundwater Level Depletion Forecasting — Western UP

AI-powered prediction & risk classification for 6 districts in Western Uttar Pradesh

## Project Overview
This project forecasts groundwater level depletion using 21 years of 
Central Ground Water Board (CGWB) quarterly well-depth data (1996-2017) 
across 6 districts: Saharanpur, Muzaffarnagar, Meerut, Bijnor, Moradabad, Bareilly.

## SDG Alignment
- SDG 6: Clean Water and Sanitation
- SDG 13: Climate Action

## AI Techniques Used
- Random Forest Regressor (lag features + seasonal index)
- Linear Trend Analysis (OLS)
- Risk Classification: Low / Moderate / High

## Key Results
| District | Trend (m/yr) | Risk |
|---|---|---|
| Saharanpur | +0.002 | Low |
| Muzaffarnagar | +0.216 | Moderate |
| Meerut | +0.125 | Moderate |
| Bijnor | +0.058 | Low |
| Moradabad | +0.134 | Moderate |
| Bareilly | +0.025 | Low |

## How to Run Dashboard
1. Download `groundwater_watch.html`
2. Double-click to open in any browser
3. No installation or internet required

## Files
- `groundwater_watch.html` — Offline dashboard
- `01_data_prep.py` — Data cleaning script
- `02_train_forecast.py` — Model training & forecasting
- `district_quarterly_clean.csv` — Cleaned dataset
- `forecast_results.json` — Model output

## Internship
1M1B AI for Sustainability Virtual Internship | IBM SkillsBuild & AICTE

**Student:** Badal Kumar | Quantum University, Roorkee
