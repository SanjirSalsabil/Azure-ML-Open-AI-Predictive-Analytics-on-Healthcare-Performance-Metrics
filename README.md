# Azure-ML-Open-AI-Predictive-Analytics-on-Healthcare-Performance-Metrics
This analysis provides specific scenarios to reduce/stabilize surgical waitlist growth for Healthcare. Key focus: Forecast the procedure increase ($10k-15k$ more/year) needed to achieve waitlist reduction targets (e.g., $25%, 50%$, or elimination within 2-5 years). Projections are broken down by Area, Facility, Specialty, and Patient Service Type

# Azure ML What / if Analysis Dashboard
![waitlist_correct_new](https://github.com/user-attachments/assets/c731949a-d001-41ea-814f-5e1fc0c88aa9)

What-if/counterfactual analysis follows:
1.	Data loading and preprocessing
2.	Exploratory data analysis (EDA) 
3.	Key metrics calculation
4.	Interactive visualization components
5.	Simulation capabilities for what-if scenarios (e.g., what if we reduce wait times, increase capacity, prioritize certain cases differently)
6.	Counterfactual analysis with visuals and dropdown parameters
Since this what/if analysis will be integrated within Azure ML Studio Notebooks, I use Python libraries for visualization and potentially ipywidgets for interactive controls.

This comprehensive framework provides:
1.	Interactive Dashboards: Real-time filtering and visualization
2.	Simulation Engine: What-if scenario modelling with multiple parameters
3.	Counterfactual Analysis: Understanding what could have been
4.	Monte Carlo Simulation: Measures Uncertainty 
5.	Automated Reporting: Executive summaries and detailed analysis
6.	ROI Analysis: Business case development for improvements
<img width="468" height="318" alt="image" src="https://github.com/user-attachments/assets/7ec167f6-ca0d-425d-ad53-5dc8f4b0d257" />


![Azure_ML_Vs_studio](https://github.com/user-attachments/assets/4e8059cb-92a4-4cab-9d1c-1ae94f3e12dc)


# Current Dataset: Healthcare waitlist data from 250,000+ surgical cases.
<img width="688" height="899" alt="dataset" src="https://github.com/user-attachments/assets/2a1ffea7-b922-42f3-87ba-743653354c28" />

Primary Data Columns for Analysis 

Days_Waiting_FROM_Decision_Date_For_Surgery_TO_Date_Of_Surgery (Avg: 128.9 days) Days_Waiting_FROM_Received_By_OR_Booking_Date_TO_Date_Of_Surgery (Avg: 112.3 days)
Case_Time_Minutes (Avg: 94.3 minutes)
Acuity_Code (I, II, III, IV priority levels)
Status 
Patient Classification Patient_Age_Group (for capacity planning by demographics) 
Emergency_Case (urgent vs. elective procedures)
Patient_Type (Day Surgery vs. Inpatient)

![Azure_ML_Vs_studio](https://github.com/user-attachments/assets/e96af3b5-e831-4514-bc62-27790f53319d)

# Wait time and Priority level breakdown
![priority_level_breakdown](https://github.com/user-attachments/assets/a9b51685-634d-4993-b702-ef53a740aca2)


# Overview of Monthly Capacity & Projected Backlog Clearance Period
<img width="788" height="340" alt="Monthly_Surgery_Target" src="https://github.com/user-attachments/assets/045d17a6-5dd0-4189-b351-d7888bf9834e" />

![Dashboard1](https://github.com/user-attachments/assets/970cab3a-3224-43a2-be0f-44e8135c110d)


# Surgeries by Specialty (Departments)
![surgeries by specialty](https://github.com/user-attachments/assets/1ee85c90-c94d-4dc7-b9bb-3d120cb483a7)

Azure AI | Machine Learning Studio (Custom Model Selection)
![Azure_ai_resource](https://github.com/user-attachments/assets/28995f11-95c1-488c-af41-0188c0b82ae0)

Features for Scenario Modelling

A. Baseline Capacity Metrics (by Facility-Specialty-RHA) 

Current Monthly Capacity Formula: Total_Cases / 12 (assuming 1-year historical data)
Average Case Duration Current average: 94.3 minutes per procedure & varies by specialty (60-154 minutes)
Current Backlog Size: Completed cases as baseline across all combinations 
•	Time period (2-5 years)
•	Additional monthly procedures (100-500)
•	Capacity increase percentage (20%-75%)
•	Backlog reduction target (50%-100%)

B. Focused on high-volume facilities - Analyzing the top 5 combinations by volume across 55 Facility-Specialty-Area combinations instead of general facility/specialty analysis

Total capacity available = Current_Monthly_Capacity × 12 × Scenario_Years 

Target Column: Additional_Monthly_Procedure
Apply to specific Facility-Specialty-Area combinations 
New capacity = Current_Monthly_Capacity + Additional_Monthly_Procedures 

Target Column: Capacity_Increase_Percent
Formula: New_Capacity = Current_Monthly_Capacity × (1 + Capacity_Increase_Percent/100)

Target Column: Backlog_Reduction_Target
Formula: Target_Cases_To_Clear = Current_Backlog × (Backlog_Reduction_Target/100)

Cases Cleared per Year Formula: (Current_Monthly_Capacity + Additional_Procedures) × 12

C. Backlog Elimination Timeline
Formula: Remaining_Backlog / Annual_Capacity_After_Increase
Resource Utilization based on Case_Time_Minutes * Staff allocation by facility and specialty


Deployment with Endpoint Creation
![Azure_endpoint](https://github.com/user-attachments/assets/7ab73256-5cb7-4bd4-ab1b-48c1d418076b)

Full Dashboard: https://claude.ai/public/artifacts/773aa7a4-3217-46ee-97df-105dac78b985 <img width="468" height="28" alt="image" src="https://github.com/user-attachments/assets/993ebec5-f00f-4a56-bbcd-75cf8159915f" />

What's causing the long waits?"
•	Booking-to-surgery delay (biggest bottleneck)
•	Case complexity (longer surgeries) correlates with longer waits
•	Facility imbalances - some at 200+ days average



