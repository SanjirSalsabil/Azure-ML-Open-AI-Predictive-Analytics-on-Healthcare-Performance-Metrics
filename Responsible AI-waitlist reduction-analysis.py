# Healthcare Waitlist Reduction - Responsible AI Toolbox Analysis
# Predict wait times and analyze counterfactual scenarios for backlog reduction

# %% [markdown]
# # Healthcare Waitlist Reduction with Responsible AI
# This notebook demonstrates using Microsoft's Responsible AI Toolbox to:
# 1. Build a model to predict surgical wait times
# 2. Analyze model fairness across different patient groups
# 3. Generate counterfactual scenarios for reducing waitlist backlog
# 4. Calculate business impact metrics (years to clear backlog, resource requirements)

# %% Cell 1: Install Required Packages
%pip install raiwidgets>=0.30.0 dice-ml interpret-community fairlearn scikit-learn xgboost lightgbm --upgrade

# %% Cell 2: Import Libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Responsible AI Libraries
from raiwidgets import ResponsibleAIDashboard
from responsibleai import RAIInsights
from interpret.ext.blackbox import TabularExplainer
from dice_ml import Data, Model
import dice_ml

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Libraries imported successfully!")

# %% Cell 3: Load and Clean Data
# Load the waitlist data
df_original = pd.read_csv("azureml://datastores/workspaceblobstore/paths/UI/2025-05-23_002002_UTC/250515_waitlist_clean.csv")

# Parse dates
date_columns = [
    'Decision_Date_For_Surgery', 'Surgery_Completed_Date',
    'Date_Referral_Received_By_Surgeon', 'Scheduled_Surgery Date'
]

for col in date_columns:
    if col in df_original.columns:
        df_original[col] = pd.to_datetime(df_original[col], format='%d/%m/%Y %H:%M', errors='coerce')

# Data Quality Check - Fix negative wait times
print("DATA QUALITY ANALYSIS:")
print(f"Original records: {len(df_original)}")

# Check for negative wait times
negative_waits = df_original['Days_Waiting_FROM_Decision_Date_For_Surgery_TO_Date_Of_Surgery'] < 0
print(f"Records with negative wait times: {negative_waits.sum()} ({negative_waits.sum()/len(df_original)*100:.1f}%)")

# Recalculate wait times from dates
df_original['Calculated_Wait_Days'] = (df_original['Surgery_Completed_Date'] - df_original['Decision_Date_For_Surgery']).dt.days

# Filter out invalid records
valid_records = (
    (df_original['Surgery_Completed_Date'] >= df_original['Decision_Date_For_Surgery']) &
    (df_original['Decision_Date_For_Surgery'].notna()) &
    (df_original['Surgery_Completed_Date'].notna()) &
    (df_original['Calculated_Wait_Days'] >= 0) &
    (df_original['Calculated_Wait_Days'] <= 1000)  # Remove extreme outliers
)

df = df_original[valid_records].copy()
df['Total_Wait_Days'] = df['Calculated_Wait_Days']

print(f"\nAfter cleaning:")
print(f"  Valid records: {len(df)} ({len(df)/len(df_original)*100:.1f}%)")
print(f"  Mean wait: {df['Total_Wait_Days'].mean():.1f} days")
print(f"  Median wait: {df['Total_Wait_Days'].median():.1f} days")

# Feature Engineering
df['Month_of_Decision'] = df['Decision_Date_For_Surgery'].dt.month
df['Year_of_Decision'] = df['Decision_Date_For_Surgery'].dt.year
df['Day_of_Week'] = df['Decision_Date_For_Surgery'].dt.dayofweek
df['Urgency_Category'] = pd.cut(df['Score'], 
                                bins=[0, 25, 50, 75, 100],
                                labels=['Low', 'Medium', 'High', 'Critical'])
df['Age_Group'] = pd.cut(df['Patient_Age'],
                        bins=[0, 18, 40, 60, 80, 100],
                        labels=['<18', '18-40', '40-60', '60-80', '80+'])

# Create Facility-Specialty-RHA grouping
df['Facility_Specialty_RHA'] = df['Facility'] + ' - ' + df['Specialty'] + ' - ' + df['RHA']

print(f"\nUnique Facility-Specialty-RHA combinations: {df['Facility_Specialty_RHA'].nunique()}")

# %% Cell 4: Prepare Features for ML Model
# Select relevant features
feature_columns = [
    'Patient_Age', 'Score', 'Case_Time_Minutes',
    'Month_of_Decision', 'Year_of_Decision', 'Day_of_Week',
    'Days_Waiting_FROM_Referral_Received_Date_TO_First_Visit_To_Surgeon',
    'Days_Waiting_FROM_First_Visit_To_Surgeon_TO_Decision_For_Surgery_Date'
]

# Categorical features to encode
categorical_features = ['Specialty', 'Facility', 'Acuity_Code', 'Patient_Type', 
                       'Anesthesia_Type', 'Cancer_Related', 'Urgency_Category']

# Create a clean dataset
df_ml = df.dropna(subset=feature_columns + ['Total_Wait_Days'] + categorical_features)

# Encode categorical variables
label_encoders = {}
for cat_feature in categorical_features:
    le = LabelEncoder()
    df_ml[f'{cat_feature}_encoded'] = le.fit_transform(df_ml[cat_feature])
    label_encoders[cat_feature] = le
    feature_columns.append(f'{cat_feature}_encoded')

# Prepare final features
X = df_ml[feature_columns]
y = df_ml['Total_Wait_Days']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# %% Cell 5: Train Multiple Models
print("Training models for wait time prediction...")

# 1. Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# 2. LightGBM (best for RAI)
lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
lgb_model.fit(X_train_scaled, y_train)
lgb_pred = lgb_model.predict(X_test_scaled)

# Model evaluation
models = {
    'Random Forest': rf_pred,
    'LightGBM': lgb_pred
}

print("\nModel Performance:")
print("-" * 60)
for name, predictions in models.items():
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"{name:20} MAE: {mae:6.1f} days | RMSE: {rmse:6.1f} | R²: {r2:.3f}")

best_model = lgb_model

# %% Cell 6: Prepare Data for RAI Dashboard
# Create feature names list
feature_names = list(X.columns)

# Prepare test data
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
test_data = df_ml.loc[X_test.index].copy()

# Add sensitive features for fairness analysis
sensitive_features = test_data[['Age_Group', 'Facility', 'Specialty']]

print(f"Test data prepared: {X_test_df.shape}")
print(f"Sensitive features: {list(sensitive_features.columns)}")

# %% Cell 7: Create Responsible AI Dashboard
print("Creating Responsible AI Dashboard...")

# Initialize RAI Insights
rai_insights = RAIInsights(
    model=best_model,
    train=pd.DataFrame(X_train_scaled, columns=feature_names),
    test=X_test_df,
    target_column='y',
    train_labels=y_train,
    test_labels=y_test,
    task_type='regression',
    categorical_features=[]
)

# Add components
print("Adding error analysis...")
rai_insights.error_analysis.add()

print("Adding model explanations...")
rai_insights.explainer.add()

print("Adding counterfactual analysis...")
rai_insights.counterfactual.add(
    total_CFs=10,
    desired_range=[30, 60]  # Target wait time range
)

print("Adding causal analysis...")
rai_insights.causal.add(treatment_features=['Case_Time_Minutes', 'Score'])

# Compute insights
print("Computing insights...")
rai_insights.compute()

# %% Cell 8: Launch RAI Dashboard
# Create and display dashboard
ResponsibleAIDashboard(rai_insights)

# %% Cell 9: Counterfactual Analysis for Decision Making
print("Generating counterfactual scenarios for waitlist reduction...")

# Prepare data for DiCE
d = dice_ml.Data(
    dataframe=pd.concat([
        pd.DataFrame(X_train_scaled, columns=feature_names),
        pd.Series(y_train, name='Total_Wait_Days')
    ], axis=1),
    continuous_features=feature_names,
    outcome_name='Total_Wait_Days'
)

# Create model wrapper
m = dice_ml.Model(model=best_model, backend='sklearn', model_type='regressor')

# Initialize DiCE
exp = dice_ml.Dice(d, m, method='random')

# Generate counterfactuals for high wait time cases
high_wait_cases = X_test_df[y_test > 180].head(5)  # Cases waiting > 6 months

counterfactual_results = []
for idx, case in high_wait_cases.iterrows():
    print(f"\nGenerating counterfactuals for case {idx} (current wait: {y_test.loc[idx]:.0f} days)")
    
    # Generate counterfactuals
    cf = exp.generate_counterfactuals(
        case.values.reshape(1, -1),
        total_CFs=5,
        desired_range=[30, 90]
    )
    
    # Store results
    if cf.cf_examples_list[0].final_cfs_df is not None:
        cf_df = cf.cf_examples_list[0].final_cfs_df
        cf_df['original_wait'] = y_test.loc[idx]
        cf_df['case_id'] = idx
        counterfactual_results.append(cf_df)

# Combine all counterfactuals
if counterfactual_results:
    all_counterfactuals = pd.concat(counterfactual_results, ignore_index=True)
    print(f"\nGenerated {len(all_counterfactuals)} counterfactual scenarios")

# %% Cell 10: Business Impact Analysis - Facility-Specialty-RHA Level
print("Calculating business impact metrics based on Facility-Specialty-RHA combinations...")

# Calculate baseline metrics per document
current_avg_wait = df['Total_Wait_Days'].mean()
avg_case_time = 94.3  # From document
total_procedures = len(df)
unique_combinations = df['Facility_Specialty_RHA'].nunique()

# Group by Facility-Specialty-RHA
fsr_metrics = df.groupby('Facility_Specialty_RHA').agg({
    'Total_Wait_Days': ['mean', 'sum', 'count'],
    'Case_Time_Minutes': 'mean'
}).round(1)

fsr_metrics.columns = ['Avg_Wait', 'Total_Wait_Days', 'Case_Count', 'Avg_Case_Time']
fsr_metrics['Monthly_Capacity'] = fsr_metrics['Case_Count'] / 12  # Assuming 1 year of data
fsr_metrics = fsr_metrics.sort_values('Case_Count', ascending=False)

print(f"\nCurrent State:")
print(f"- Total procedures: {total_procedures:,}")
print(f"- Unique Facility-Specialty-RHA combinations: {unique_combinations}")
print(f"- Average wait: {current_avg_wait:.1f} days")
print(f"- Average case time: {avg_case_time:.1f} minutes")

# High-priority combinations
high_volume_fsr = fsr_metrics.head(5)
print(f"\nTop 5 combinations by volume:")
for idx, (combo, row) in enumerate(high_volume_fsr.iterrows()):
    print(f"{idx+1}. {combo}: {row['Case_Count']:.0f} cases, {row['Avg_Wait']:.1f} days avg wait")

# %% Cell 11: Scenario Analysis with Document Parameters
# Scenario definitions from document
scenarios = {
    'Conservative - 5 Years': {
        'years': 5,
        'additional_monthly_procedures': 150,
        'capacity_increase': 20,
        'backlog_reduction_target': 75
    },
    'Conservative - 3 Years': {
        'years': 3,
        'additional_monthly_procedures': 200,
        'capacity_increase': 25,
        'backlog_reduction_target': 50
    },
    'Aggressive - 3 Years': {
        'years': 3,
        'additional_monthly_procedures': 350,
        'capacity_increase': 50,
        'backlog_reduction_target': 80
    },
    'Aggressive - 2 Years': {
        'years': 2,
        'additional_monthly_procedures': 500,
        'capacity_increase': 75,
        'backlog_reduction_target': 100
    }
}

# Calculate impact for each scenario
results = []
for scenario_name, params in scenarios.items():
    # For top 5 facilities combined
    current_monthly_total = high_volume_fsr['Monthly_Capacity'].sum()
    current_backlog_total = high_volume_fsr['Case_Count'].sum()
    
    # New capacity calculations
    capacity_from_increase = current_monthly_total * (params['capacity_increase']/100)
    total_new_monthly = current_monthly_total + capacity_from_increase + params['additional_monthly_procedures']
    
    # Calculate outcomes
    target_cases = current_backlog_total * (params['backlog_reduction_target']/100)
    annual_new_capacity = (total_new_monthly - current_monthly_total) * 12
    total_capacity_over_period = total_new_monthly * 12 * params['years']
    
    # Time to clear target
    if annual_new_capacity > 0:
        years_to_target = target_cases / annual_new_capacity
        feasible = years_to_target <= params['years']
    else:
        years_to_target = float('inf')
        feasible = False
    
    # New average wait time estimate
    capacity_factor = current_monthly_total / total_new_monthly
    new_avg_wait = current_avg_wait * capacity_factor
    
    results.append({
        'Scenario': scenario_name,
        'Time Horizon (years)': params['years'],
        'Additional Monthly Procedures': params['additional_monthly_procedures'],
        'Capacity Increase (%)': params['capacity_increase'],
        'Backlog Target (%)': params['backlog_reduction_target'],
        'Cases to Clear': int(target_cases),
        'New Avg Wait (days)': new_avg_wait,
        'Years to Achieve Target': round(years_to_target, 1) if years_to_target != float('inf') else 'N/A',
        'Feasible': 'Yes' if feasible else 'No'
    })

impact_df = pd.DataFrame(results)
print("\nSCENARIO ANALYSIS RESULTS:")
print(impact_df.to_string(index=False))

# %% Cell 12: Visualization of Impact Analysis
# Create comprehensive impact visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Years to Clear Target Backlog',
        'New Average Wait Time',
        'Total Cases to Clear',
        'Feasibility by Scenario'
    ),
    specs=[[{'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'scatter'}]]
)

# 1. Years to clear backlog
valid_years = impact_df[impact_df['Years to Achieve Target'] != 'N/A']['Years to Achieve Target'].astype(float)
valid_scenarios = impact_df[impact_df['Years to Achieve Target'] != 'N/A']['Scenario']

fig.add_trace(
    go.Bar(
        x=valid_scenarios,
        y=valid_years,
        text=[f"{y:.1f} years" for y in valid_years],
        textposition='auto',
        marker_color=['green' if y <= impact_df.loc[impact_df['Scenario'] == s, 'Time Horizon (years)'].values[0] 
                     else 'red' for s, y in zip(valid_scenarios, valid_years)]
    ),
    row=1, col=1
)

# 2. New average wait time
fig.add_trace(
    go.Bar(
        x=impact_df['Scenario'],
        y=impact_df['New Avg Wait (days)'],
        text=[f"{w:.0f} days" for w in impact_df['New Avg Wait (days)']],
        textposition='auto',
        marker_color='lightblue'
    ),
    row=1, col=2
)

# Add current average line
fig.add_hline(y=current_avg_wait, line_dash="dash", line_color="red", 
              annotation_text=f"Current: {current_avg_wait:.0f} days",
              row=1, col=2)

# 3. Cases to clear
fig.add_trace(
    go.Bar(
        x=impact_df['Scenario'],
        y=impact_df['Cases to Clear'],
        text=[f"{c:,}" for c in impact_df['Cases to Clear']],
        textposition='auto',
        marker_color='orange'
    ),
    row=2, col=1
)

# 4. Feasibility timeline
colors = ['green' if f == 'Yes' else 'red' for f in impact_df['Feasible']]
fig.add_trace(
    go.Scatter(
        x=impact_df['Time Horizon (years)'],
        y=impact_df['Capacity Increase (%)'],
        mode='markers+text',
        text=impact_df['Scenario'],
        textposition='top center',
        marker=dict(size=20, color=colors),
        showlegend=False
    ),
    row=2, col=2
)

fig.update_xaxes(title_text="Time Horizon (years)", row=2, col=2)
fig.update_yaxes(title_text="Capacity Increase (%)", row=2, col=2)

fig.update_layout(
    height=800,
    title_text="Healthcare Waitlist Reduction Scenario Analysis",
    showlegend=False
)

fig.show()

# %% Cell 13: ROI and Resource Analysis
print("\nRESOURCE REQUIREMENTS AND ROI ANALYSIS:")
print("="*70)

for _, scenario in impact_df.iterrows():
    print(f"\n{scenario['Scenario']}:")
    
    # Resource requirements
    additional_monthly = scenario['Additional Monthly Procedures']
    additional_annual = additional_monthly * 12
    
    # Staffing estimates (1 surgeon per 20 procedures/month)
    additional_surgeons = additional_monthly / 20
    
    # OR time requirements (using 94.3 min average)
    additional_or_hours_monthly = (additional_monthly * 94.3) / 60
    additional_or_days_monthly = additional_or_hours_monthly / 8  # 8-hour OR days
    
    print(f"  Resource Requirements:")
    print(f"    - Additional surgeons needed: {additional_surgeons:.1f}")
    print(f"    - Additional OR days/month: {additional_or_days_monthly:.1f}")
    print(f"    - Additional cases/year: {additional_annual:,}")
    
    # Cost estimation
    capacity_cost = scenario['Capacity Increase (%)'] * 100000  # $100k per 1% capacity
    procedure_cost = additional_annual * 2000  # $2k per additional procedure
    total_cost = capacity_cost + procedure_cost
    
    # Benefit calculation
    wait_reduction = current_avg_wait - scenario['New Avg Wait (days)']
    value_per_day_saved = 150  # Patient benefit value
    annual_benefit = wait_reduction * total_procedures * value_per_day_saved
    
    roi = (annual_benefit - total_cost) / total_cost * 100 if total_cost > 0 else 0
    payback = total_cost / annual_benefit if annual_benefit > 0 else float('inf')
    
    print(f"  Financial Analysis:")
    print(f"    - Investment required: ${total_cost:,.0f}")
    print(f"    - Annual benefit: ${annual_benefit:,.0f}")
    print(f"    - ROI: {roi:.0f}%")
    print(f"    - Payback period: {payback:.1f} years" if payback != float('inf') else "    - Payback period: N/A")

# %% Cell 14: Generate Actionable Recommendations
print("\n" + "="*70)
print("ACTIONABLE RECOMMENDATIONS FOR WAITLIST REDUCTION")
print("="*70)

# Feature importance from model
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n1. KEY DRIVERS OF WAIT TIMES:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   - {row['feature']}: {row['importance']:.3f}")

# High-wait facilities
high_wait_facilities = fsr_metrics[fsr_metrics['Avg_Wait'] > 200].head(5)
if len(high_wait_facilities) > 0:
    print("\n2. CRITICAL FACILITIES (>200 days average wait):")
    for facility in high_wait_facilities.index:
        print(f"   - {facility}: {high_wait_facilities.loc[facility, 'Avg_Wait']:.0f} days")

print("\n3. RECOMMENDED IMPLEMENTATION PLAN:")
print("   Phase 1 (Immediate - 3 months):")
print("   - Implement booking process improvements")
print("   - Redistribute cases from high-wait to low-wait facilities")
print("   - Focus on top 5 high-volume combinations")

print("\n   Phase 2 (3-12 months):")
print("   - Add capacity per 'Conservative - 3 Years' scenario")
print("   - Target 200 additional procedures/month")
print("   - Achieve 25% capacity increase")

print("\n   Phase 3 (1-2 years):")
print("   - Scale to 'Aggressive' scenario if Phase 2 successful")
print("   - Implement ML-based dynamic scheduling")
print("   - Target 80% backlog reduction")

print("\n4. SUCCESS METRICS:")
print("   - Reduce average wait from {:.0f} to <90 days".format(current_avg_wait))
print("   - Clear 50% of current backlog within 3 years")
print("   - Maintain wait times <60 days for urgent cases")
print("   - Achieve positive ROI within 12 months")

print("\n" + "="*70)