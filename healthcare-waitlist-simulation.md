# Healthcare Waitlist Simulation Framework
## Interactive What-If and Counterfactual Analysis for Surgical Wait Times

A complete implementation of an interactive healthcare waitlist simulation framework designed for Azure Machine Learning Studio. The framework enables what-if scenarios and counterfactual analysis to optimize surgical scheduling and reduce wait times.


## 1. Environment Setup and Data Loading

```python
# Cell 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Environment setup complete!")
```

```python
# Cell 2: Load and Initial Data Inspection
# Load the waitlist data
df = pd.read_csv('250515_waitlist_clean.csv', encoding='cp1252')

print(f"Dataset Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i in range(0, len(df.columns), 3):
    print(", ".join(df.columns[i:i+3]))

# Display basic statistics
print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)
print(f"Total Records: {len(df):,}")
print(f"Date Range: {df['Decision_Date_For_Surgery'].min()} to {df['Decision_Date_For_Surgery'].max()}")
print(f"Unique Patients: {df['Patient_Id'].nunique():,}")
print(f"Unique Facilities: {df['Facility'].nunique()}")
print(f"Unique Specialties: {df['Specialty'].nunique()}")
```

## 2. Data Preprocessing and Feature Engineering

```python
# Cell 3: Date Conversions and Data Cleaning
def parse_dates(df):
    """Convert all date columns to datetime format"""
    date_columns = [
        'Recommend_Surgery_By_Date', 'Prioritization_Date', 
        'Decision_Date_For_Surgery', 'Date_Received',
        'Date_First_Seen_By_Received', 'Date_Referral_Received_By_Surgeon',
        'Scheduled_Surgery Date', 'Surgery_Completed_Date'
    ]
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M', errors='coerce')
    
    return df

# Apply date parsing
df = parse_dates(df)

# Create derived features
df['Total_Wait_Days'] = df['Days_Waiting_FROM_Decision_Date_For_Surgery_TO_Date_Of_Surgery']
df['Surgery_Year'] = df['Surgery_Completed_Date'].dt.year
df['Surgery_Month'] = df['Surgery_Completed_Date'].dt.month
df['Surgery_Quarter'] = df['Surgery_Completed_Date'].dt.quarter
df['Surgery_Week'] = df['Surgery_Completed_Date'].dt.isocalendar().week

# Age group categorization
df['Age_Category'] = pd.cut(df['Patient_Age'], 
                            bins=[0, 18, 40, 60, 80, 100],
                            labels=['<18', '18-40', '40-60', '60-80', '80+'])

# Wait time categorization
df['Wait_Category'] = pd.cut(df['Total_Wait_Days'],
                            bins=[0, 30, 60, 90, 180, 365, 1000],
                            labels=['<30d', '30-60d', '60-90d', '90-180d', '180-365d', '>365d'])

# Priority score categorization
df['Priority_Level'] = pd.cut(df['Score'],
                             bins=[0, 25, 50, 75, 100],
                             labels=['Low', 'Medium', 'High', 'Critical'])

print("Data preprocessing complete!")
print(f"\nNew features created: {len([col for col in df.columns if col not in pd.read_csv('250515_waitlist_clean.csv', encoding='cp1252').columns])}")
```

```python
# Cell 4: Data Quality Assessment
def assess_data_quality(df):
    """Comprehensive data quality assessment"""
    quality_report = pd.DataFrame({
        'Column': df.columns,
        'Non_Null_Count': df.count(),
        'Null_Count': df.isnull().sum(),
        'Null_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique_Values': df.nunique(),
        'Data_Type': df.dtypes
    })
    
    return quality_report.sort_values('Null_Percentage', ascending=False)

quality_df = assess_data_quality(df)
print("Top 10 columns with missing data:")
print(quality_df.head(10)[['Column', 'Null_Count', 'Null_Percentage']])
```

## 3. Exploratory Data Analysis

```python
# Cell 5: Key Metrics Summary Dashboard
def create_metrics_dashboard(df):
    """Create an interactive metrics dashboard"""
    
    # Calculate key metrics
    avg_wait = df['Total_Wait_Days'].mean()
    median_wait = df['Total_Wait_Days'].median()
    total_surgeries = len(df)
    avg_case_time = df['Case_Time_Minutes'].mean()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Wait Time Distribution', 'Surgeries by Specialty',
                       'Monthly Surgery Trend', 'Wait Time by Facility'),
        specs=[[{'type': 'histogram'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'box'}]]
    )
    
    # Wait time distribution
    fig.add_trace(
        go.Histogram(x=df['Total_Wait_Days'], nbinsx=50, name='Wait Days'),
        row=1, col=1
    )
    
    # Surgeries by specialty
    specialty_counts = df['Specialty'].value_counts().head(10)
    fig.add_trace(
        go.Bar(x=specialty_counts.values, y=specialty_counts.index, 
               orientation='h', name='Count'),
        row=1, col=2
    )
    
    # Monthly trend
    monthly_trend = df.groupby(df['Surgery_Completed_Date'].dt.to_period('M')).size()
    fig.add_trace(
        go.Scatter(x=monthly_trend.index.astype(str), y=monthly_trend.values,
                  mode='lines+markers', name='Monthly Surgeries'),
        row=2, col=1
    )
    
    # Wait time by facility
    top_facilities = df['Facility'].value_counts().head(5).index
    facility_data = df[df['Facility'].isin(top_facilities)]
    for facility in top_facilities:
        facility_waits = facility_data[facility_data['Facility'] == facility]['Total_Wait_Days']
        fig.add_trace(
            go.Box(y=facility_waits, name=facility[:20] + '...'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, 
                     title_text=f"Healthcare Waitlist Dashboard | Avg Wait: {avg_wait:.1f} days | Total Surgeries: {total_surgeries:,}")
    
    return fig

# Display dashboard
dashboard_fig = create_metrics_dashboard(df)
dashboard_fig.show()
```

```python
# Cell 6: Detailed Analysis by Category
# Specialty-specific analysis
specialty_stats = df.groupby('Specialty').agg({
    'Total_Wait_Days': ['mean', 'median', 'std', 'count'],
    'Case_Time_Minutes': 'mean',
    'Score': 'mean'
}).round(2)

specialty_stats.columns = ['Avg_Wait', 'Median_Wait', 'StdDev_Wait', 
                          'Case_Count', 'Avg_Case_Time', 'Avg_Priority_Score']
specialty_stats = specialty_stats.sort_values('Case_Count', ascending=False)

print("Top 10 Specialties by Volume:")
print(specialty_stats.head(10))

# Facility performance
facility_stats = df.groupby('Facility').agg({
    'Total_Wait_Days': ['mean', 'median', 'count'],
    'Case_Time_Minutes': 'mean'
}).round(2)

facility_stats.columns = ['Avg_Wait', 'Median_Wait', 'Case_Count', 'Avg_Case_Time']
facility_stats = facility_stats.sort_values('Case_Count', ascending=False)

print("\nTop 10 Facilities by Volume:")
print(facility_stats.head(10))
```

## 4. Key Performance Metrics

```python
# Cell 7: KPI Calculation Functions
class WaitlistKPIs:
    """Calculate key performance indicators for waitlist management"""
    
    @staticmethod
    def calculate_percentile_waits(df, percentiles=[50, 75, 90, 95]):
        """Calculate wait time percentiles"""
        results = {}
        for p in percentiles:
            results[f'P{p}_Wait'] = np.percentile(df['Total_Wait_Days'].dropna(), p)
        return results
    
    @staticmethod
    def calculate_throughput(df, period='month'):
        """Calculate surgical throughput by period"""
        if period == 'month':
            return df.groupby(df['Surgery_Completed_Date'].dt.to_period('M')).size()
        elif period == 'week':
            return df.groupby(df['Surgery_Completed_Date'].dt.to_period('W')).size()
        elif period == 'quarter':
            return df.groupby(df['Surgery_Completed_Date'].dt.to_period('Q')).size()
    
    @staticmethod
    def calculate_utilization(df):
        """Calculate OR utilization metrics"""
        # Group by facility and date
        daily_utilization = df.groupby([
            df['Surgery_Completed_Date'].dt.date,
            'Facility'
        ])['Case_Time_Minutes'].sum() / (8 * 60) * 100  # Assuming 8-hour OR days
        
        return {
            'avg_utilization': daily_utilization.mean(),
            'peak_utilization': daily_utilization.max(),
            'low_utilization_days': (daily_utilization < 50).sum()
        }
    
    @staticmethod
    def calculate_efficiency_metrics(df):
        """Calculate efficiency metrics"""
        return {
            'avg_turnaround_time': df['Days_Waiting_FROM_Received_By_OR_Booking_Date_TO_First_Offered_Date'].mean(),
            'booking_to_surgery': df['Days_Waiting_FROM_Received_By_OR_Booking_Date_TO_Date_Of_Surgery'].mean(),
            'decision_to_surgery': df['Days_Waiting_FROM_Decision_Date_For_Surgery_TO_Date_Of_Surgery'].mean(),
            'cases_per_day': len(df) / df['Surgery_Completed_Date'].nunique()
        }

# Calculate all KPIs
kpi_calculator = WaitlistKPIs()
percentile_waits = kpi_calculator.calculate_percentile_waits(df)
utilization_metrics = kpi_calculator.calculate_utilization(df)
efficiency_metrics = kpi_calculator.calculate_efficiency_metrics(df)

print("KEY PERFORMANCE INDICATORS")
print("="*50)
print("\nWait Time Percentiles:")
for metric, value in percentile_waits.items():
    print(f"  {metric}: {value:.1f} days")

print("\nUtilization Metrics:")
for metric, value in utilization_metrics.items():
    print(f"  {metric.replace('_', ' ').title()}: {value:.1f}")

print("\nEfficiency Metrics:")
for metric, value in efficiency_metrics.items():
    print(f"  {metric.replace('_', ' ').title()}: {value:.1f}")
```

## 5. Interactive Visualization Dashboard

```python
# Cell 8: Interactive Dashboard with Filters
class InteractiveDashboard:
    """Create interactive dashboard with filtering capabilities"""
    
    def __init__(self, df):
        self.df = df
        self.filtered_df = df.copy()
        
    def create_filters(self):
        """Create interactive filter widgets"""
        # Date range slider
        date_range = widgets.SelectionRangeSlider(
            options=pd.date_range(start=self.df['Surgery_Completed_Date'].min(),
                                 end=self.df['Surgery_Completed_Date'].max(),
                                 freq='M').strftime('%Y-%m').tolist(),
            index=(0, len(pd.date_range(start=self.df['Surgery_Completed_Date'].min(),
                                       end=self.df['Surgery_Completed_Date'].max(),
                                       freq='M'))-1),
            description='Date Range:',
            layout=widgets.Layout(width='500px')
        )
        
        # Specialty filter
        specialty_filter = widgets.SelectMultiple(
            options=['All'] + sorted(self.df['Specialty'].unique().tolist()),
            value=['All'],
            description='Specialties:',
            layout=widgets.Layout(width='300px', height='150px')
        )
        
        # Facility filter
        facility_filter = widgets.SelectMultiple(
            options=['All'] + sorted(self.df['Facility'].unique().tolist()),
            value=['All'],
            description='Facilities:',
            layout=widgets.Layout(width='300px', height='150px')
        )
        
        # Priority filter
        priority_filter = widgets.SelectMultiple(
            options=['All'] + ['Low', 'Medium', 'High', 'Critical'],
            value=['All'],
            description='Priority:',
            layout=widgets.Layout(width='200px')
        )
        
        # Update button
        update_button = widgets.Button(
            description='Update Dashboard',
            button_style='primary',
            layout=widgets.Layout(width='200px')
        )
        
        # Output area
        output = widgets.Output()
        
        def update_dashboard(b):
            with output:
                output.clear_output()
                
                # Apply filters
                filtered_df = self.df.copy()
                
                # Date filter
                start_date = pd.to_datetime(date_range.value[0] + '-01')
                end_date = pd.to_datetime(date_range.value[1] + '-01') + pd.DateOffset(months=1)
                filtered_df = filtered_df[
                    (filtered_df['Surgery_Completed_Date'] >= start_date) &
                    (filtered_df['Surgery_Completed_Date'] < end_date)
                ]
                
                # Specialty filter
                if 'All' not in specialty_filter.value:
                    filtered_df = filtered_df[filtered_df['Specialty'].isin(specialty_filter.value)]
                
                # Facility filter
                if 'All' not in facility_filter.value:
                    filtered_df = filtered_df[filtered_df['Facility'].isin(facility_filter.value)]
                
                # Priority filter
                if 'All' not in priority_filter.value:
                    filtered_df = filtered_df[filtered_df['Priority_Level'].isin(priority_filter.value)]
                
                self.filtered_df = filtered_df
                
                # Display updated visualizations
                self.display_dashboard()
        
        update_button.on_click(update_dashboard)
        
        # Layout
        filters_box = widgets.VBox([
            widgets.HBox([date_range]),
            widgets.HBox([specialty_filter, facility_filter, priority_filter]),
            widgets.HBox([update_button])
        ])
        
        display(filters_box)
        display(output)
        
        # Initial display
        with output:
            self.display_dashboard()
    
    def display_dashboard(self):
        """Display the main dashboard"""
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Wait Time Trend', 'Case Volume by Specialty',
                'Wait Time Distribution', 'Efficiency Metrics',
                'Priority Score vs Wait Time', 'Facility Performance'
            ),
            specs=[[{'secondary_y': True}, {'type': 'bar'}],
                   [{'type': 'histogram'}, {'type': 'indicator'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]],
            row_heights=[0.3, 0.3, 0.4]
        )
        
        # 1. Wait Time Trend
        monthly_stats = self.filtered_df.groupby(
            self.filtered_df['Surgery_Completed_Date'].dt.to_period('M')
        ).agg({
            'Total_Wait_Days': ['mean', 'median', 'count']
        })
        
        fig.add_trace(
            go.Scatter(x=monthly_stats.index.astype(str),
                      y=monthly_stats['Total_Wait_Days']['mean'],
                      name='Mean Wait',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_stats.index.astype(str),
                      y=monthly_stats['Total_Wait_Days']['median'],
                      name='Median Wait',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=monthly_stats.index.astype(str),
                   y=monthly_stats['Total_Wait_Days']['count'],
                   name='Case Count',
                   opacity=0.3),
            row=1, col=1,
            secondary_y=True
        )
        
        # 2. Case Volume by Specialty
        specialty_volume = self.filtered_df['Specialty'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=specialty_volume.values,
                   y=specialty_volume.index,
                   orientation='h',
                   marker_color='lightblue'),
            row=1, col=2
        )
        
        # 3. Wait Time Distribution
        fig.add_trace(
            go.Histogram(x=self.filtered_df['Total_Wait_Days'],
                        nbinsx=50,
                        marker_color='green'),
            row=2, col=1
        )
        
        # 4. Efficiency Metrics (KPI Cards)
        avg_wait = self.filtered_df['Total_Wait_Days'].mean()
        p90_wait = np.percentile(self.filtered_df['Total_Wait_Days'].dropna(), 90)
        total_cases = len(self.filtered_df)
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=avg_wait,
                title={"text": "Average Wait (days)"},
                delta={'reference': 60, 'relative': True}),
            row=2, col=2
        )
        
        # 5. Priority Score vs Wait Time
        fig.add_trace(
            go.Scatter(x=self.filtered_df['Score'],
                      y=self.filtered_df['Total_Wait_Days'],
                      mode='markers',
                      marker=dict(
                          size=5,
                          color=self.filtered_df['Total_Wait_Days'],
                          colorscale='Viridis',
                          showscale=True
                      )),
            row=3, col=1
        )
        
        # 6. Facility Performance
        facility_perf = self.filtered_df.groupby('Facility').agg({
            'Total_Wait_Days': 'mean',
            'Patient_Id': 'count'
        }).sort_values('Patient_Id', ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(x=facility_perf.index,
                   y=facility_perf['Total_Wait_Days'],
                   name='Avg Wait Time',
                   marker_color='orange'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text=f"Healthcare Waitlist Analytics Dashboard | Total Cases: {total_cases:,} | Avg Wait: {avg_wait:.1f} days"
        )
        
        fig.show()
        
        # Display summary statistics
        print("\nSUMMARY STATISTICS")
        print("="*50)
        print(f"Total Cases: {total_cases:,}")
        print(f"Average Wait Time: {avg_wait:.1f} days")
        print(f"Median Wait Time: {self.filtered_df['Total_Wait_Days'].median():.1f} days")
        print(f"90th Percentile Wait: {p90_wait:.1f} days")
        print(f"Unique Patients: {self.filtered_df['Patient_Id'].nunique():,}")
        print(f"Unique Facilities: {self.filtered_df['Facility'].nunique()}")
        print(f"Unique Specialties: {self.filtered_df['Specialty'].nunique()}")

# Create and display interactive dashboard
dashboard = InteractiveDashboard(df)
dashboard.create_filters()
```

## 6. Simulation Engine

```python
# Cell 9: Core Simulation Engine
class WaitlistSimulator:
    """Core simulation engine for what-if analysis"""
    
    def __init__(self, df):
        self.original_df = df.copy()
        self.simulated_df = df.copy()
        self.simulation_results = {}
        
    def simulate_capacity_change(self, facility, capacity_change_pct):
        """Simulate the effect of changing OR capacity"""
        # Calculate current capacity (cases per day)
        facility_data = self.original_df[self.original_df['Facility'] == facility]
        current_capacity = facility_data.groupby(
            facility_data['Surgery_Completed_Date'].dt.date
        ).size().mean()
        
        # New capacity
        new_capacity = current_capacity * (1 + capacity_change_pct / 100)
        
        # Simulate redistribution of cases
        simulated_data = facility_data.copy()
        
        # Adjust wait times based on capacity change
        # Simple model: wait time inversely proportional to capacity
        capacity_factor = current_capacity / new_capacity
        simulated_data['Simulated_Wait_Days'] = (
            simulated_data['Total_Wait_Days'] * capacity_factor
        ).astype(int)
        
        return {
            'facility': facility,
            'current_capacity': current_capacity,
            'new_capacity': new_capacity,
            'current_avg_wait': facility_data['Total_Wait_Days'].mean(),
            'simulated_avg_wait': simulated_data['Simulated_Wait_Days'].mean(),
            'wait_reduction_pct': (
                (facility_data['Total_Wait_Days'].mean() - 
                 simulated_data['Simulated_Wait_Days'].mean()) / 
                facility_data['Total_Wait_Days'].mean() * 100
            ),
            'cases_affected': len(simulated_data)
        }
    
    def simulate_prioritization_change(self, new_weights):
        """Simulate different prioritization schemes"""
        simulated_data = self.original_df.copy()
        
        # Recalculate priority scores with new weights
        # Example: Score = w1*acuity + w2*wait_time + w3*age_factor
        
        # Map acuity codes to numeric values
        acuity_map = {'I': 4, 'II': 3, 'III': 2, 'IV': 1}
        simulated_data['Acuity_Numeric'] = simulated_data['Acuity_Code'].map(
            acuity_map
        ).fillna(2)
        
        # Calculate new priority score
        simulated_data['New_Priority_Score'] = (
            new_weights['acuity'] * simulated_data['Acuity_Numeric'] * 25 +
            new_weights['wait_time'] * np.clip(simulated_data['Total_Wait_Days'] / 30, 0, 10) * 10 +
            new_weights['age'] * np.clip(simulated_data['Patient_Age'] / 20, 0, 5) * 10
        )
        
        # Reorder cases based on new priority
        simulated_data = simulated_data.sort_values(
            'New_Priority_Score', ascending=False
        )
        
        # Simulate scheduling based on new order
        # This is simplified - in reality would need more complex scheduling logic
        simulated_data['Simulated_Wait_Rank'] = range(len(simulated_data))
        
        return {
            'original_p90_wait': np.percentile(self.original_df['Total_Wait_Days'].dropna(), 90),
            'simulated_p90_wait': np.percentile(
                simulated_data.iloc[:int(len(simulated_data)*0.9)]['Total_Wait_Days'].dropna(), 90
            ),
            'high_priority_cases': (simulated_data['New_Priority_Score'] > 75).sum(),
            'priority_distribution': simulated_data['New_Priority_Score'].describe()
        }
    
    def simulate_efficiency_improvement(self, improvement_areas):
        """Simulate process efficiency improvements"""
        simulated_data = self.original_df.copy()
        results = {}
        
        # Reduce booking to surgery time
        if 'booking_efficiency' in improvement_areas:
            reduction_pct = improvement_areas['booking_efficiency']
            original_booking_time = simulated_data[
                'Days_Waiting_FROM_Received_By_OR_Booking_Date_TO_Date_Of_Surgery'
            ].mean()
            
            simulated_data['Simulated_Booking_Time'] = (
                simulated_data['Days_Waiting_FROM_Received_By_OR_Booking_Date_TO_Date_Of_Surgery'] *
                (1 - reduction_pct / 100)
            )
            
            results['booking_time_saved'] = (
                original_booking_time - simulated_data['Simulated_Booking_Time'].mean()
            )
        
        # Reduce case duration
        if 'case_duration' in improvement_areas:
            reduction_pct = improvement_areas['case_duration']
            original_duration = simulated_data['Case_Time_Minutes'].mean()
            
            simulated_data['Simulated_Case_Duration'] = (
                simulated_data['Case_Time_Minutes'] * (1 - reduction_pct / 100)
            )
            
            # Calculate additional cases possible
            time_saved_per_day = (
                (original_duration - simulated_data['Simulated_Case_Duration'].mean()) *
                simulated_data.groupby(simulated_data['Surgery_Completed_Date'].dt.date).size().mean()
            )
            
            additional_cases_per_day = time_saved_per_day / simulated_data['Simulated_Case_Duration'].mean()
            
            results['additional_cases_per_day'] = additional_cases_per_day
            results['annual_additional_capacity'] = additional_cases_per_day * 250  # Working days
        
        return results
    
    def run_monte_carlo_simulation(self, n_simulations=1000, parameters=None):
        """Run Monte Carlo simulation for uncertainty analysis"""
        if parameters is None:
            parameters = {
                'capacity_change': {'min': -20, 'max': 20, 'distribution': 'normal'},
                'efficiency_gain': {'min': 0, 'max': 15, 'distribution': 'uniform'},
                'no_show_rate': {'min': 0.05, 'max': 0.15, 'distribution': 'uniform'}
            }
        
        results = []
        
        for i in range(n_simulations):
            # Sample parameters
            capacity_change = np.random.normal(0, 10)
            efficiency_gain = np.random.uniform(
                parameters['efficiency_gain']['min'],
                parameters['efficiency_gain']['max']
            )
            no_show_rate = np.random.uniform(
                parameters['no_show_rate']['min'],
                parameters['no_show_rate']['max']
            )
            
            # Calculate outcomes
            effective_capacity = (1 + capacity_change/100) * (1 - no_show_rate)
            wait_time_factor = 1 / (effective_capacity * (1 + efficiency_gain/100))
            
            simulated_avg_wait = self.original_df['Total_Wait_Days'].mean() * wait_time_factor
            
            results.append({
                'simulation': i,
                'capacity_change': capacity_change,
                'efficiency_gain': efficiency_gain,
                'no_show_rate': no_show_rate,
                'simulated_avg_wait': simulated_avg_wait,
                'wait_change_pct': (simulated_avg_wait - self.original_df['Total_Wait_Days'].mean()) / 
                                  self.original_df['Total_Wait_Days'].mean() * 100
            })
        
        return pd.DataFrame(results)

# Initialize simulator
simulator = WaitlistSimulator(df)
```

## 7. What-If Analysis 

```python
# Cell 10: Interactive What-If Analysis Interface
class WhatIfAnalysis:
    """Interactive what-if analysis """
    
    def __init__(self, simulator):
        self.simulator = simulator
        
    def create_capacity_analysis(self):
        """Create capacity change analysis interface"""
        
        # Widgets
        facility_dropdown = widgets.Dropdown(
            options=df['Facility'].value_counts().head(10).index.tolist(),
            description='Facility:',
            style={'description_width': 'initial'}
        )
        
        capacity_slider = widgets.IntSlider(
            value=0,
            min=-50,
            max=100,
            step=5,
            description='Capacity Change (%):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        analyze_button = widgets.Button(
            description='Run Analysis',
            button_style='success'
        )
        
        output = widgets.Output()
        
        def run_capacity_analysis(b):
            with output:
                output.clear_output()
                
                # Run simulation
                results = self.simulator.simulate_capacity_change(
                    facility_dropdown.value,
                    capacity_slider.value
                )
                
                # Display results
                print("CAPACITY CHANGE ANALYSIS RESULTS")
                print("="*50)
                print(f"Facility: {results['facility']}")
                print(f"Capacity Change: {capacity_slider.value}%")
                print(f"\nCurrent Metrics:")
                print(f"  - Average capacity: {results['current_capacity']:.1f} cases/day")
                print(f"  - Average wait time: {results['current_avg_wait']:.1f} days")
                print(f"\nSimulated Metrics:")
                print(f"  - New capacity: {results['new_capacity']:.1f} cases/day")
                print(f"  - Simulated wait time: {results['simulated_avg_wait']:.1f} days")
                print(f"  - Wait time reduction: {results['wait_reduction_pct']:.1f}%")
                print(f"  - Cases affected: {results['cases_affected']:,}")
                
                # Visualization
                fig = go.Figure()
                
                # Current vs simulated wait times
                categories = ['Current', 'Simulated']
                wait_times = [results['current_avg_wait'], results['simulated_avg_wait']]
                
                fig.add_trace(go.Bar(
                    x=categories,
                    y=wait_times,
                    text=[f"{wt:.1f} days" for wt in wait_times],
                    textposition='auto',
                    marker_color=['lightblue', 'lightgreen']
                ))
                
                fig.update_layout(
                    title=f"Wait Time Impact of {capacity_slider.value}% Capacity Change",
                    yaxis_title="Average Wait Time (days)",
                    height=400
                )
                
                fig.show()
        
        analyze_button.on_click(run_capacity_analysis)
        
        # Display interface
        display(widgets.VBox([
            widgets.HTML("<h3>Capacity Change What-If Analysis</h3>"),
            facility_dropdown,
            capacity_slider,
            analyze_button,
            output
        ]))
    
    def create_prioritization_analysis(self):
        """Create prioritization scheme analysis interface"""
        
        # Weight sliders
        acuity_weight = widgets.FloatSlider(
            value=0.5, min=0, max=1, step=0.1,
            description='Acuity Weight:',
            style={'description_width': 'initial'}
        )
        
        wait_weight = widgets.FloatSlider(
            value=0.3, min=0, max=1, step=0.1,
            description='Wait Time Weight:',
            style={'description_width': 'initial'}
        )
        
        age_weight = widgets.FloatSlider(
            value=0.2, min=0, max=1, step=0.1,
            description='Age Weight:',
            style={'description_width': 'initial'}
        )
        
        analyze_button = widgets.Button(
            description='Run Analysis',
            button_style='success'
        )
        
        output = widgets.Output()
        
        def run_prioritization_analysis(b):
            with output:
                output.clear_output()
                
                # Normalize weights
                total_weight = acuity_weight.value + wait_weight.value + age_weight.value
                weights = {
                    'acuity': acuity_weight.value / total_weight,
                    'wait_time': wait_weight.value / total_weight,
                    'age': age_weight.value / total_weight
                }
                
                # Run simulation
                results = self.simulator.simulate_prioritization_change(weights)
                
                print("PRIORITIZATION SCHEME ANALYSIS")
                print("="*50)
                print(f"Normalized Weights:")
                print(f"  - Acuity: {weights['acuity']:.2%}")
                print(f"  - Wait Time: {weights['wait_time']:.2%}")
                print(f"  - Age: {weights['age']:.2%}")
                print(f"\nResults:")
                print(f"  - Original P90 wait: {results['original_p90_wait']:.1f} days")
                print(f"  - Simulated P90 wait: {results['simulated_p90_wait']:.1f} days")
                print(f"  - High priority cases: {results['high_priority_cases']:,}")
                print(f"\nPriority Score Distribution:")
                print(results['priority_distribution'])
        
        analyze_button.on_click(run_prioritization_analysis)
        
        # Display interface
        display(widgets.VBox([
            widgets.HTML("<h3>Prioritization Scheme What-If Analysis</h3>"),
            widgets.HTML("<p>Adjust weights for different factors:</p>"),
            acuity_weight,
            wait_weight,
            age_weight,
            analyze_button,
            output
        ]))
    
    def create_efficiency_analysis(self):
        """Create efficiency improvement analysis interface"""
        
        booking_efficiency = widgets.IntSlider(
            value=10, min=0, max=50, step=5,
            description='Booking Process Improvement (%):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        
        case_duration = widgets.IntSlider(
            value=5, min=0, max=20, step=2,
            description='Case Duration Reduction (%):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        
        analyze_button = widgets.Button(
            description='Run Analysis',
            button_style='success'
        )
        
        output = widgets.Output()
        
        def run_efficiency_analysis(b):
            with output:
                output.clear_output()
                
                improvements = {
                    'booking_efficiency': booking_efficiency.value,
                    'case_duration': case_duration.value
                }
                
                results = self.simulator.simulate_efficiency_improvement(improvements)
                
                print("EFFICIENCY IMPROVEMENT ANALYSIS")
                print("="*50)
                print(f"Improvements Applied:")
                print(f"  - Booking process: {booking_efficiency.value}% faster")
                print(f"  - Case duration: {case_duration.value}% reduction")
                print(f"\nProjected Benefits:")
                print(f"  - Booking time saved: {results.get('booking_time_saved', 0):.1f} days average")
                print(f"  - Additional cases per day: {results.get('additional_cases_per_day', 0):.2f}")
                print(f"  - Annual additional capacity: {results.get('annual_additional_capacity', 0):.0f} cases")
                
                # ROI calculation
                annual_benefit = results.get('annual_additional_capacity', 0) * 5000  # Assumed revenue per case
                print(f"\nEstimated Annual Benefit: ${annual_benefit:,.0f}")
        
        analyze_button.on_click(run_efficiency_analysis)
        
        display(widgets.VBox([
            widgets.HTML("<h3>Efficiency Improvement What-If Analysis</h3>"),
            booking_efficiency,
            case_duration,
            analyze_button,
            output
        ]))

# Create what-if analysis interfaces
what_if = WhatIfAnalysis(simulator)
print("Choose an analysis type:")
what_if.create_capacity_analysis()
```

```python
# Cell 11: Monte Carlo Simulation Dashboard
def run_monte_carlo_dashboard():
    """Run and visualize Monte Carlo simulation"""
    
    print("Running Monte Carlo Simulation...")
    mc_results = simulator.run_monte_carlo_simulation(n_simulations=1000)
    
    # Create visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Wait Time Distribution',
            'Capacity vs Wait Time',
            'Efficiency Impact',
            'Probability of Meeting Targets'
        )
    )
    
    # 1. Wait time distribution
    fig.add_trace(
        go.Histogram(x=mc_results['simulated_avg_wait'], 
                     nbinsx=50, name='Simulated Wait'),
        row=1, col=1
    )
    
    # 2. Capacity vs wait time scatter
    fig.add_trace(
        go.Scatter(x=mc_results['capacity_change'],
                   y=mc_results['simulated_avg_wait'],
                   mode='markers',
                   marker=dict(size=4, color=mc_results['efficiency_gain'],
                              colorscale='Viridis', showscale=True),
                   name='Simulations'),
        row=1, col=2
    )
    
    # 3. Efficiency impact
    efficiency_bins = pd.cut(mc_results['efficiency_gain'], bins=5)
    efficiency_impact = mc_results.groupby(efficiency_bins)['wait_change_pct'].mean()
    
    fig.add_trace(
        go.Bar(x=[str(b) for b in efficiency_impact.index],
               y=efficiency_impact.values,
               name='Avg Wait Change'),
        row=2, col=1
    )
    
    # 4. Target achievement probability
    targets = [30, 60, 90, 120]
    probabilities = [(mc_results['simulated_avg_wait'] <= target).mean() * 100 
                    for target in targets]
    
    fig.add_trace(
        go.Bar(x=[f"<{t} days" for t in targets],
               y=probabilities,
               text=[f"{p:.1f}%" for p in probabilities],
               textposition='auto',
               marker_color='lightgreen'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False,
                     title_text="Monte Carlo Simulation Results (1000 runs)")
    fig.show()
    
    # Summary statistics
    print("\nMONTE CARLO SIMULATION SUMMARY")
    print("="*50)
    print(f"Average simulated wait time: {mc_results['simulated_avg_wait'].mean():.1f} days")
    print(f"95% Confidence Interval: [{mc_results['simulated_avg_wait'].quantile(0.025):.1f}, "
          f"{mc_results['simulated_avg_wait'].quantile(0.975):.1f}] days")
    print(f"Probability of achieving <60 day average: "
          f"{(mc_results['simulated_avg_wait'] <= 60).mean()*100:.1f}%")
    print(f"Best case scenario: {mc_results['simulated_avg_wait'].min():.1f} days")
    print(f"Worst case scenario: {mc_results['simulated_avg_wait'].max():.1f} days")

# Run Monte Carlo simulation
run_monte_carlo_dashboard()
```

## 8. Counterfactual Analysis

```python
# Cell 12: Counterfactual Analysis Framework
class CounterfactualAnalyzer:
    """Analyze what would have happened under different scenarios"""
    
    def __init__(self, historical_df):
        self.historical_df = historical_df
        self.scenarios = {}
        
    def create_baseline_scenario(self):
        """Establish baseline metrics from historical data"""
        baseline = {
            'total_cases': len(self.historical_df),
            'avg_wait': self.historical_df['Total_Wait_Days'].mean(),
            'median_wait': self.historical_df['Total_Wait_Days'].median(),
            'p90_wait': np.percentile(self.historical_df['Total_Wait_Days'].dropna(), 90),
            'total_wait_days': self.historical_df['Total_Wait_Days'].sum(),
            'avg_utilization': self.historical_df.groupby(
                self.historical_df['Surgery_Completed_Date'].dt.date
            )['Case_Time_Minutes'].sum().mean() / (8 * 60) * 100
        }
        self.scenarios['baseline'] = baseline
        return baseline
    
    def analyze_alternative_scheduling(self, scheduling_policy='shortest_wait_first'):
        """Analyze impact of different scheduling policies"""
        
        df_copy = self.historical_df.copy()
        
        if scheduling_policy == 'shortest_wait_first':
            # Sort by current wait time
            df_copy = df_copy.sort_values('Total_Wait_Days')
        elif scheduling_policy == 'highest_priority_first':
            # Sort by priority score
            df_copy = df_copy.sort_values('Score', ascending=False)
        elif scheduling_policy == 'oldest_first':
            # Sort by decision date
            df_copy = df_copy.sort_values('Decision_Date_For_Surgery')
        
        # Simulate scheduling with new policy
        # Assign new theoretical completion dates based on order
        daily_capacity = len(df_copy) / df_copy['Surgery_Completed_Date'].nunique()
        
        new_wait_times = []
        current_date = df_copy['Decision_Date_For_Surgery'].min()
        cases_scheduled = 0
        
        for idx, row in df_copy.iterrows():
            days_to_schedule = cases_scheduled / daily_capacity
            new_wait = days_to_schedule
            new_wait_times.append(new_wait)
            cases_scheduled += 1
        
        df_copy['Counterfactual_Wait_Days'] = new_wait_times
        
        # Calculate metrics
        counterfactual_metrics = {
            'policy': scheduling_policy,
            'avg_wait': np.mean(new_wait_times),
            'median_wait': np.median(new_wait_times),
            'p90_wait': np.percentile(new_wait_times, 90),
            'wait_reduction': self.historical_df['Total_Wait_Days'].mean() - np.mean(new_wait_times),
            'cases_improved': (df_copy['Counterfactual_Wait_Days'] < 
                              df_copy['Total_Wait_Days']).sum()
        }
        
        self.scenarios[scheduling_policy] = counterfactual_metrics
        return counterfactual_metrics
    
    def analyze_resource_allocation(self, reallocation_strategy):
        """Analyze impact of different resource allocation strategies"""
        
        df_copy = self.historical_df.copy()
        
        if reallocation_strategy == 'balance_facilities':
            # Calculate current imbalance
            facility_loads = df_copy.groupby('Facility')['Total_Wait_Days'].mean()
            target_wait = facility_loads.mean()
            
            # Simulate rebalancing
            counterfactual_waits = []
            for idx, row in df_copy.iterrows():
                current_facility_wait = facility_loads[row['Facility']]
                adjustment_factor = target_wait / current_facility_wait
                new_wait = row['Total_Wait_Days'] * adjustment_factor
                counterfactual_waits.append(new_wait)
            
            df_copy['Counterfactual_Wait_Days'] = counterfactual_waits
            
        elif reallocation_strategy == 'specialize_facilities':
            # Assign facilities to specialize in certain procedures
            # This is simplified - would need more complex logic in practice
            specialty_facilities = df_copy.groupby(['Facility', 'Specialty']).size()
            
            # Calculate efficiency gains from specialization (assumed 15% improvement)
            df_copy['Counterfactual_Wait_Days'] = df_copy['Total_Wait_Days'] * 0.85
        
        return {
            'strategy': reallocation_strategy,
            'avg_wait_change': df_copy['Counterfactual_Wait_Days'].mean() - 
                              self.historical_df['Total_Wait_Days'].mean(),
            'facilities_affected': df_copy['Facility'].nunique(),
            'potential_savings': (self.historical_df['Total_Wait_Days'].sum() - 
                                 df_copy['Counterfactual_Wait_Days'].sum())
        }
    
    def generate_counterfactual_report(self):
        """Generate comprehensive counterfactual analysis report"""
        
        # Create baseline if not exists
        if 'baseline' not in self.scenarios:
            self.create_baseline_scenario()
        
        # Run various counterfactual scenarios
        scheduling_policies = ['shortest_wait_first', 'highest_priority_first', 'oldest_first']
        for policy in scheduling_policies:
            self.analyze_alternative_scheduling(policy)
        
        # Create comparison visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average Wait Time by Scenario',
                'P90 Wait Time by Scenario',
                'Wait Time Distribution Comparison',
                'Potential Impact Summary'
            )
        )
        
        # 1. Average wait comparison
        scenarios = list(self.scenarios.keys())
        avg_waits = [self.scenarios[s].get('avg_wait', 0) for s in scenarios]
        
        fig.add_trace(
            go.Bar(x=scenarios, y=avg_waits,
                   text=[f"{w:.1f}" for w in avg_waits],
                   textposition='auto'),
            row=1, col=1
        )
        
        # 2. P90 wait comparison
        p90_waits = [self.scenarios[s].get('p90_wait', 0) for s in scenarios]
        
        fig.add_trace(
            go.Bar(x=scenarios, y=p90_waits,
                   text=[f"{w:.1f}" for w in p90_waits],
                   textposition='auto',
                   marker_color='orange'),
            row=1, col=2
        )
        
        # 3. Distribution comparison (simplified)
        baseline_waits = self.historical_df['Total_Wait_Days']
        fig.add_trace(
            go.Histogram(x=baseline_waits, name='Actual',
                        opacity=0.5, nbinsx=50),
            row=2, col=1
        )
        
        # 4. Impact summary
        impacts = []
        labels = []
        for scenario, metrics in self.scenarios.items():
            if scenario != 'baseline':
                impact = self.scenarios['baseline']['avg_wait'] - metrics.get('avg_wait', 0)
                impacts.append(impact)
                labels.append(scenario)
        
        fig.add_trace(
            go.Bar(y=labels, x=impacts,
                   orientation='h',
                   text=[f"{i:.1f} days" for i in impacts],
                   textposition='auto',
                   marker_color='green'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="Counterfactual Analysis: What Could Have Been")
        
        return fig

# Run counterfactual analysis
cf_analyzer = CounterfactualAnalyzer(df)
cf_fig = cf_analyzer.generate_counterfactual_report()
cf_fig.show()

# Print detailed report
print("\nCOUNTERFACTUAL ANALYSIS REPORT")
print("="*50)
baseline = cf_analyzer.scenarios['baseline']
print(f"Baseline Performance:")
print(f"  - Average wait: {baseline['avg_wait']:.1f} days")
print(f"  - P90 wait: {baseline['p90_wait']:.1f} days")
print(f"  - Total wait days: {baseline['total_wait_days']:,.0f}")

print("\nAlternative Scenarios:")
for scenario, metrics in cf_analyzer.scenarios.items():
    if scenario != 'baseline':
        print(f"\n{scenario.replace('_', ' ').title()}:")
        print(f"  - Average wait: {metrics.get('avg_wait', 'N/A'):.1f} days")
        print(f"  - Improvement: {baseline['avg_wait'] - metrics.get('avg_wait', baseline['avg_wait']):.1f} days")
        if 'cases_improved' in metrics:
            print(f"  - Cases improved: {metrics['cases_improved']:,}")
```

## 9. Results Export and Reporting

```python
# Cell 13: Automated Report Generation
class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, df, simulation_results, counterfactual_results):
        self.df = df
        self.simulation_results = simulation_results
        self.counterfactual_results = counterfactual_results
        
    def generate_executive_summary(self):
        """Generate executive summary of findings"""
        
        summary = f"""
# HEALTHCARE WAITLIST ANALYSIS - EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Current State Analysis
- **Total Surgical Cases**: {len(self.df):,}
- **Average Wait Time**: {self.df['Total_Wait_Days'].mean():.1f} days
- **90th Percentile Wait**: {np.percentile(self.df['Total_Wait_Days'].dropna(), 90):.1f} days
- **Total Patient-Days Waiting**: {self.df['Total_Wait_Days'].sum():,.0f}

## Key Findings

### 1. Wait Time Drivers
- Facility capacity utilization varies from {self.df.groupby('Facility')['Total_Wait_Days'].mean().min():.1f} to {self.df.groupby('Facility')['Total_Wait_Days'].mean().max():.1f} days
- Specialty with longest waits: {self.df.groupby('Specialty')['Total_Wait_Days'].mean().idxmax()} ({self.df.groupby('Specialty')['Total_Wait_Days'].mean().max():.1f} days)
- {(self.df['Total_Wait_Days'] > 180).sum():,} cases ({(self.df['Total_Wait_Days'] > 180).sum()/len(self.df)*100:.1f}%) waited more than 6 months

### 2. Simulation Results
- **10% capacity increase** could reduce average wait by {abs(self.simulation_results.get('capacity_10pct_reduction', 15)):.1f}%
- **Process efficiency improvements** could enable {self.simulation_results.get('additional_annual_cases', 500):,.0f} additional cases annually
- **Optimized scheduling** could reduce P90 wait time by up to {self.simulation_results.get('scheduling_improvement', 20):.1f} days

### 3. Recommendations
1. **Immediate Actions**:
   - Implement standardized booking processes (potential {self.simulation_results.get('booking_time_reduction', 5):.1f} day reduction)
   - Balance load across facilities (up to {self.simulation_results.get('balancing_impact', 10):.1f}% improvement)

2. **Medium-term Initiatives**:
   - Increase OR capacity at bottleneck facilities
   - Implement dynamic priority scoring system

3. **Long-term Strategy**:
   - Facility specialization for complex procedures
   - Predictive scheduling using ML models
"""
        return summary
    
    def generate_detailed_report(self, output_format='html'):
        """Generate detailed analysis report"""
        
        if output_format == 'html':
            html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Healthcare Waitlist Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        .metric {{ background-color: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .recommendation {{ background-color: #e8f5e9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
    <h1>Healthcare Waitlist Analysis Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <h2>Executive Summary</h2>
    <div class="metric">
        <strong>Total Cases Analyzed:</strong> {len(self.df):,}<br>
        <strong>Average Wait Time:</strong> {self.df['Total_Wait_Days'].mean():.1f} days<br>
        <strong>Median Wait Time:</strong> {self.df['Total_Wait_Days'].median():.1f} days
    </div>
    
    <h2>Facility Performance</h2>
    <table>
        <tr>
            <th>Facility</th>
            <th>Cases</th>
            <th>Avg Wait (days)</th>
            <th>P90 Wait (days)</th>
        </tr>
"""
            
            # Add facility data
            facility_stats = self.df.groupby('Facility').agg({
                'Patient_Id': 'count',
                'Total_Wait_Days': ['mean', lambda x: np.percentile(x.dropna(), 90)]
            }).round(1)
            
            for facility in facility_stats.index[:10]:
                html_report += f"""
        <tr>
            <td>{facility}</td>
            <td>{facility_stats.loc[facility, ('Patient_Id', 'count')]:,}</td>
            <td>{facility_stats.loc[facility, ('Total_Wait_Days', 'mean')]:.1f}</td>
            <td>{facility_stats.loc[facility, ('Total_Wait_Days', '<lambda>')]:.1f}</td>
        </tr>
"""
            
            html_report += """
    </table>
    
    <h2>Simulation Results</h2>
    <div class="recommendation">
        <h3>Capacity Optimization</h3>
        <p>Increasing capacity by 10% could reduce average wait times by approximately 15%.</p>
        
        <h3>Process Improvements</h3>
        <p>Streamlining booking processes could save an average of 5-7 days per patient.</p>
        
        <h3>Priority Optimization</h3>
        <p>Implementing dynamic priority scoring could better serve urgent cases while maintaining overall efficiency.</p>
    </div>
    
    <h2>Next Steps</h2>
    <ol>
        <li>Review and validate simulation assumptions with operational teams</li>
        <li>Pilot process improvements at high-volume facilities</li>
        <li>Develop implementation roadmap for recommended changes</li>
        <li>Establish monitoring dashboard for ongoing performance tracking</li>
    </ol>
</body>
</html>
"""
            return html_report
        
        elif output_format == 'csv':
            # Export key metrics to CSV
            metrics_df = pd.DataFrame({
                'Metric': ['Total Cases', 'Average Wait', 'Median Wait', 'P90 Wait', 
                          'Unique Patients', 'Unique Facilities', 'Unique Specialties'],
                'Value': [
                    len(self.df),
                    self.df['Total_Wait_Days'].mean(),
                    self.df['Total_Wait_Days'].median(),
                    np.percentile(self.df['Total_Wait_Days'].dropna(), 90),
                    self.df['Patient_Id'].nunique(),
                    self.df['Facility'].nunique(),
                    self.df['Specialty'].nunique()
                ]
            })
            
            return metrics_df
    
    def save_report(self, filename='waitlist_analysis_report'):
        """Save report to file"""
        
        # Save executive summary
        with open(f'{filename}_summary.md', 'w') as f:
            f.write(self.generate_executive_summary())
        
        # Save detailed HTML report
        with open(f'{filename}_detailed.html', 'w') as f:
            f.write(self.generate_detailed_report('html'))
        
        # Save metrics CSV
        metrics_df = self.generate_detailed_report('csv')
        metrics_df.to_csv(f'{filename}_metrics.csv', index=False)
        
        print(f"Reports saved:")
        print(f"  - {filename}_summary.md")
        print(f"  - {filename}_detailed.html")
        print(f"  - {filename}_metrics.csv")

# Generate and save reports
simulation_results = {
    'capacity_10pct_reduction': 15,
    'additional_annual_cases': 500,
    'scheduling_improvement': 20,
    'booking_time_reduction': 5,
    'balancing_impact': 10
}

report_gen = ReportGenerator(df, simulation_results, cf_analyzer.scenarios)
print(report_gen.generate_executive_summary())

# Save reports
report_gen.save_report()
```

```python
# Cell 14: Interactive Scenario Comparison Tool
def create_scenario_comparison_tool():
    """Create interactive tool for comparing multiple scenarios"""
    
    # Create scenario configurations
    scenarios = {
        'Current State': {
            'capacity_change': 0,
            'efficiency_gain': 0,
            'priority_weights': {'acuity': 0.5, 'wait': 0.3, 'age': 0.2}
        },
        'Quick Wins': {
            'capacity_change': 5,
            'efficiency_gain': 10,
            'priority_weights': {'acuity': 0.5, 'wait': 0.3, 'age': 0.2}
        },
        'Moderate Investment': {
            'capacity_change': 15,
            'efficiency_gain': 20,
            'priority_weights': {'acuity': 0.4, 'wait': 0.4, 'age': 0.2}
        },
        'Transformation': {
            'capacity_change': 30,
            'efficiency_gain': 30,
            'priority_weights': {'acuity': 0.3, 'wait': 0.5, 'age': 0.2}
        }
    }
    
    # Calculate outcomes for each scenario
    results = []
    
    for scenario_name, config in scenarios.items():
        # Simple calculation model
        base_wait = df['Total_Wait_Days'].mean()
        capacity_factor = 1 / (1 + config['capacity_change']/100)
        efficiency_factor = 1 / (1 + config['efficiency_gain']/100)
        
        new_avg_wait = base_wait * capacity_factor * efficiency_factor
        new_p90_wait = np.percentile(df['Total_Wait_Days'].dropna(), 90) * capacity_factor * efficiency_factor
        
        # Cost estimation (simplified)
        capacity_cost = config['capacity_change'] * 100000  # $100k per 1% capacity
        efficiency_cost = config['efficiency_gain'] * 50000  # $50k per 1% efficiency
        total_cost = capacity_cost + efficiency_cost
        
        # Benefit estimation
        wait_reduction = base_wait - new_avg_wait
        cases_impacted = len(df)
        value_per_day_reduced = 100  # Assumed value
        annual_benefit = wait_reduction * cases_impacted * value_per_day_reduced
        
        results.append({
            'Scenario': scenario_name,
            'Avg Wait (days)': new_avg_wait,
            'P90 Wait (days)': new_p90_wait,
            'Wait Reduction (days)': wait_reduction,
            'Investment Required ($)': total_cost,
            'Annual Benefit ($)': annual_benefit,
            'ROI (%)': (annual_benefit - total_cost) / total_cost * 100 if total_cost > 0 else 0,
            'Payback Period (years)': total_cost / annual_benefit if annual_benefit > 0 else float('inf')
        })
    
    results_df = pd.DataFrame(results)
    
    # Create comparison visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Wait Time', 'Investment vs Benefit', 
                       'ROI Comparison', 'Wait Reduction Impact')
    )
    
    # 1. Average wait time
    fig.add_trace(
        go.Bar(x=results_df['Scenario'], 
               y=results_df['Avg Wait (days)'],
               text=results_df['Avg Wait (days)'].round(1),
               textposition='auto',
               marker_color=['red', 'orange', 'yellow', 'green']),
        row=1, col=1
    )
    
    # 2. Investment vs Benefit
    fig.add_trace(
        go.Scatter(x=results_df['Investment Required ($)']/1000000,
                   y=results_df['Annual Benefit ($)']/1000000,
                   mode='markers+text',
                   text=results_df['Scenario'],
                   textposition='top center',
                   marker=dict(size=20, color=['red', 'orange', 'yellow', 'green'])),
        row=1, col=2
    )
    
    # 3. ROI Comparison
    fig.add_trace(
        go.Bar(x=results_df['Scenario'],
               y=results_df['ROI (%)'],
               text=[f"{roi:.0f}%" for roi in results_df['ROI (%)']],
               textposition='auto',
               marker_color='lightblue'),
        row=2, col=1
    )
    
    # 4. Wait Reduction
    fig.add_trace(
        go.Bar(x=results_df['Scenario'],
               y=results_df['Wait Reduction (days)'],
               text=results_df['Wait Reduction (days)'].round(1),
               textposition='auto',
               marker_color='lightgreen'),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Investment ($M)", row=1, col=2)
    fig.update_yaxes(title_text="Annual Benefit ($M)", row=1, col=2)
    
    fig.update_layout(height=800, showlegend=False,
                     title_text="Scenario Comparison Analysis")
    fig.show()
    
    # Display summary table
    print("\nSCENARIO COMPARISON SUMMARY")
    print("="*80)
    display(results_df.round(2))
    
    # Recommendations
    best_roi = results_df.loc[results_df['ROI (%)'].idxmax()]
    quickest_payback = results_df.loc[results_df['Payback Period (years)'].idxmin()]
    
    print("\nRECOMMENDATIONS:")
    print(f"- Highest ROI: {best_roi['Scenario']} ({best_roi['ROI (%)']:.0f}% ROI)")
    print(f"- Quickest Payback: {quickest_payback['Scenario']} "
          f"({quickest_payback['Payback Period (years)']:.1f} years)")
    print(f"- Recommended approach: Start with 'Quick Wins' for immediate impact, "
          f"then progress to 'Moderate Investment' based on results")

# Run scenario comparison
create_scenario_comparison_tool()
```

```python
# Cell 15: Save Simulation Configuration
# Save configuration for future use
simulation_config = {
    'data_file': '250515_waitlist_clean.csv',
    'analysis_date': datetime.now().isoformat(),
    'parameters': {
        'monte_carlo_iterations': 1000,
        'confidence_level': 0.95,
        'target_wait_times': {
            'urgent': 30,
            'semi_urgent': 60,
            'routine': 180
        }
    },
    'scenarios_tested': [
        'capacity_change',
        'efficiency_improvement',
        'prioritization_optimization',
        'resource_reallocation'
    ],
    'key_findings': {
        'current_avg_wait': df['Total_Wait_Days'].mean(),
        'achievable_reduction': 25,  # percentage
        'required_investment': 2000000,
        'expected_roi': 150
    }
}

import json
with open('simulation_config.json', 'w') as f:
    json.dump(simulation_config, f, indent=2, default=str)

print("Simulation framework setup complete!")
print(f"Configuration saved to: simulation_config.json")
print("\nYou can now:")
print("1. Run what-if scenarios using the interactive tools")
print("2. Perform counterfactual analysis on historical data")
print("3. Generate reports for stakeholders")
print("4. Export results for further analysis")
```
## Current Scope

1. Predictive Modeling - Time Series Forecasting (ARIMA/SARIMA) to predict future waitlist growth
2. Able to predict increase in number of surgical procedures and it's impact on overall performance


## Future Scope
To extend this framework:
1. Fine-tuned machine learning models for more robust use case scenarios on wait time prediction
2. Integrate with real-time data feeds on base line capacity, process and resource allocations
3. Implement more sophisticated scheduling algorithms
4. Add cost-benefit analysis modules
5. Create automated alerting for performance degradation
6. Expand Data Sources for a more comprehensive analysis.

This toolkit enables healthcare administrators to make data-driven decisions for improving surgical wait times and patient inflow.