
# ATL Flight Intelligence

Flight Departure Delay Prediction Using LightGBM

## Overview

ATL Flight Intelligence is a Streamlit-based prototype
developed as part of an undergraduate Information Systems
thesis.

The project evaluates the contribution of historical delay
features to flight departure delay prediction at
Hartsfield–Jackson Atlanta International Airport (ATL).

Predictions are designed for a two-hour pre-departure
horizon (T−2h).

## Research Scope

- Airport: Hartsfield–Jackson Atlanta International Airport
- Period: January–December 2024
- Modeling records: 338,500
- Target: Departure delay of at least 15 minutes
- Algorithm: LightGBM
- Validation: Chronological train/validation/test split
- Explainability: SHAP

## Modeling Scenarios

### Baseline V1
Seven scheduled-flight features:
- month
- day_of_week
- scheduled_dep_hour
- op_unique_carrier
- dest
- distance
- crs_elapsed_time

### Enhanced V1
Baseline features plus three historical delay rates:
- hist_delay_rate_carrier
- hist_delay_rate_dest
- hist_delay_rate_carrier_dest

Historical features use a walk-forward approach under
the study's T−2h information-availability assumptions.

## Main Results

Independent test period: November–December 2024
Test records: 56,936

| Metric | Baseline V1 | Enhanced V1 |
|---|---:|---:|
| Average Precision | 0.282596 | 0.278441 |
| Recall (Delayed) | 0.442988 | 0.440365 |
| ROC-AUC | 0.627177 | 0.622433 |

The addition of three historical features did not improve
Average Precision on the independent test set.

An additional V2 experiment explored rolling historical
windows through internal temporal validation. Its results
should not be interpreted as independent test performance.

## Streamlit Application

The deployed application uses the Baseline V1 model.

Main functionality:
- Flight schedule and route inputs
- Departure delay probability
- On-Time / Delayed classification
- Local SHAP explanations

Decision threshold: 0.2420

## Running Locally

1. Install Python and the dependencies.
2. Open a terminal in the project folder.
3. Run:

   pip install -r requirements.txt
   streamlit run app.py

The application also requires the trained model and
associated inference artifacts.

## Dataset

Source:
https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024

The dataset is based on Bureau of Transportation
Statistics (BTS) flight records.

The complete research dataset is not included in this
repository.

## Limitations

This is a retrospective research prototype, not a
real-time operational flight delay prediction system.

The dataset does not provide complete timestamps
indicating when operational information was published.
Consequently, information availability at T−2h is based
on methodological assumptions.

Results are limited to the selected airport and period.

## Deployment

Streamlit application URL:
[Add the actual URL after successful deployment]

## Research Context

Undergraduate Thesis
Information Systems
Universitas Multimedia Nusantara

## Author

Alenzo Leksana
