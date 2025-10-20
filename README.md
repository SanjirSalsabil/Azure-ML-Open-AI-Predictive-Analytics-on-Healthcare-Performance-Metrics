# Azure-ML-Open-AI-Predictive-Analytics-on-Healthcare-Performance-Metrics
This analysis provides specific scenarios to reduce/stabilize surgical waitlist growth for Healthcare. Key focus: Forecast the procedure increase ($10k-15k$ more/year) needed to achieve waitlist reduction targets (e.g., 25%, 50%, or elimination within 2-5 years). Projections are broken down by Area, Facility, Specialty, and Patient Service Type

# What / if Analysis 
![waitlist_correct_new](https://github.com/user-attachments/assets/c731949a-d001-41ea-814f-5e1fc0c88aa9)

Sample Dashboard: https://claude.ai/public/artifacts/773aa7a4-3217-46ee-97df-105dac78b985 

# Features for Scenario Modelling

A. Baseline Capacity Metrics (by Facility-Specialty-Area) 

Current Monthly Capacity Formula: Total_Cases / 12 (assuming 1-year historical data)
Average Case Duration Current average: 94.3 minutes per procedure & varies by specialty (60-154 minutes)
Current Backlog Size: Completed cases as baseline across all combinations 
•	Time period (2-5 years)
•	Additional monthly procedures (100-500)
•	Capacity increase percentage (20%-75%)
•	Backlog reduction target (50%-100%)

B. Focused on high-volume facilities - Analyzing the top 5 combinations by volume across 55 Facility-Specialty-Area combinations instead of general facility/specialty analysis

Total capacity available = Current_Monthly_Capacity × 12 × Scenario_Years,
New capacity = Current_Monthly_Capacity + Additional_Monthly_Procedures,
Target_Cases_To_Clear = Current_Backlog × (Backlog_Reduction_Target/100),
ases Cleared per Year: (Current_Monthly_Capacity + Additional_Procedures) × 12,
Backlog Elimination Timeline: Remaining_Backlog / Annual_Capacity_After_Increase,
Resource Utilization based on Case_Time_Minutes * Staff allocation by facility and specialty.

# Current Dataset: Healthcare waitlist data from 250,000+ surgical cases.
<img width="688" height="899" alt="dataset" src="https://github.com/user-attachments/assets/2a1ffea7-b922-42f3-87ba-743653354c28" />

![Azure_ML_Vs_studio](https://github.com/user-attachments/assets/e96af3b5-e831-4514-bc62-27790f53319d)

# Wait time and Priority level breakdown
![priority_level_breakdown](https://github.com/user-attachments/assets/a9b51685-634d-4993-b702-ef53a740aca2)

The Monthly Surgery Trend below shows only completed surgeries/scheduled/queued surgeries (not incorporating future cases):

<img width="346" height="145" alt="image" src="https://github.com/user-attachments/assets/63e54317-6b69-446e-87a4-0196f5b14c19" />

# Overview of Monthly Capacity & Projected Backlog Clearance Period
<img width="788" height="340" alt="Monthly_Surgery_Target" src="https://github.com/user-attachments/assets/045d17a6-5dd0-4189-b351-d7888bf9834e" />

<img width="483" height="172" alt="image" src="https://github.com/user-attachments/assets/c821cabb-a020-4a09-93a7-8526427fc620" />

A compact reference table at the top showing all key metrics for each zone:
o	Current Backlog
o	Baseline Capacity
o	Average Inflow
o	Extra Needed per Month
o	Total Required Monthly
o	Adjusted Required Monthly

![Dashboard1](https://github.com/user-attachments/assets/970cab3a-3224-43a2-be0f-44e8135c110d)

# Surgeries by Specialty (Departments)
![surgeries by specialty](https://github.com/user-attachments/assets/1ee85c90-c94d-4dc7-b9bb-3d120cb483a7)

Azure AI | Machine Learning Studio (Custom Model Selection)
![Azure_ai_resource](https://github.com/user-attachments/assets/28995f11-95c1-488c-af41-0188c0b82ae0)

Deployment with Endpoint Creation
![Azure_endpoint](https://github.com/user-attachments/assets/7ab73256-5cb7-4bd4-ab1b-48c1d418076b)

Azure ML Surgical Operations Predictive Models
MODEL 1: BACKLOG PREDICTION BY SPECIALTY/FACILITY
Target Variable: Target_Backlog - Number of patients waiting for surgery (queued + scheduled status)
Technique: XGBoost Regression with synthetic time-series data generation
Strategy:
•	Creates training data by projecting current backlogs forward/backward using net monthly flow (inflow - completions)
•	Groups predictions by specialty and facility
•	Uses 12 months of synthetic historical data for training
Key Features:
•	Current backlog count
•	Average monthly completed surgeries
•	Average monthly patient inflow
•	Net monthly flow (inflow - completions)
•	Encoded specialty and facility
•	Time features (year, month, quarter)
•	Months offset from current date
Results:
•	MAE: 5.19 patients (average prediction error)
•	R²: 0.998 (model explains 99.8% of variance - excellent fit)
•	Current total backlog: 12,776 patients
•	6-month forecast shows decreasing trend (-861 patients by month 6)
MODEL 2: PATIENT INFLOW PREDICTION
Target Variable: Daily_Inflow - Number of new patients added to surgical waitlist per day
Technique: Gradient Boosting Regression with time-series features
Strategy:
•	Uses historical daily patient arrival patterns
•	Incorporates lag features and rolling statistics for time-series forecasting
Key Features:
•	Temporal features (day of week, month, year, quarter)
•	Lag features (1, 7, 14, 30 days)
•	Rolling averages and standard deviations (7, 14, 30-day windows)
Results Interpretation:
•	MAE: 9.18 patients/day - Predictions are off by ~9 patients daily on average
•	R²: 0.921 - Model explains 92.1% of variance (excellent performance)
•	High R² indicates patient inflow follows predictable patterns



Our surgical backlog model uses the historical relationship between inflow, completions, and backlog to make predictions. The synthetic training data it creates assumes these relationships stay consistent:
# The model projects future backlogs using historical net flow patterns
projected_backlog = current_backlog + (net_flow * months_offset)


Model Choice Rationale
XGBoost (Backlog): Handles non-linear relationships between features well, robust to outliers, good for tabular data with mixed feature types.
LightGBM (Wait Time): Faster training than XGBoost, better for larger datasets, handles categorical features efficiently. Chosen because wait time data has more records (completed surgeries).
Gradient Boosting (Inflow): Best for time-series with trend/seasonal patterns. Captures complex temporal dependencies through lag features.



What's causing the long waits?"
•	Booking-to-surgery delay (biggest bottleneck)
•	Case complexity (longer surgeries) correlates with longer waits
•	Facility imbalances - some at 200+ days average

Current State
The model shows we currently have 12,776 patients waiting for surgery across all specialties and facilities. This represents patients who are either queued (waiting to be scheduled) or already scheduled but haven't had their surgery yet.
The Predicted Trend
Looking at the six-month forecast, we can see a consistent downward trend in the backlog. Each month shows a reduction in the total number of waiting patients, with the changes becoming progressively larger:
The reduction starts modestly at 175 patients in the first month, but by month six, we're seeing a reduction of 861 patients from the current level. This acceleration in backlog reduction suggests that the system is becoming more efficient at processing patients over time.
What's Driving This Pattern?
This declining backlog occurs when the rate of surgeries being completed exceeds the rate of new patients being added to the waitlist. Think of it like a bathtub - if water is draining out faster than it's flowing in, the water level drops. The model has learned from historical patterns that this net outflow tends to increase over the coming months.The forecast presents an optimistic outlook for the healthcare system. The model predicts that if current operational trends continue, the surgical backlog will decrease by approximately 6.7% (from 12,776 to 11,915 patients) over the next six months. This represents 861 fewer patients waiting for surgery.
However, it's important to remember that this prediction assumes that current conditions remain stable. Factors like seasonal variations in patient arrivals, changes in surgical capacity, staff availability, or unexpected events (like we saw with COVID-19) could alter this trajectory. The high R² value (0.998) suggests the model is very confident in these predictions based on historical patterns, but real-world healthcare systems can be subject to sudden changes that historical data might not capture.
This information would be valuable for hospital administrators to plan resources, allocate surgical time slots, and potentially communicate wait time improvements to patients and stakeholders.




