# Interactive Dashboard with Filters
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import pandas as pd


#Derived Features:  

# 1. Calculated_Wait_Days - Surgery_Completed_Date - Recommend_Surgery_By_Date 
# 2. Total_Wait_Days - Copy of Calculated_Wait_Days (replaces original column) 
# 3. Surgery_Year - Extracted from Surgery_Completed_Date 
# 4. Surgery_Month - Extracted from Surgery_Completed_Date   
# 5. Surgery_Quarter - Extracted from Surgery_Completed_Date 
# 6. Surgery_Week - ISO week number from Surgery_Completed_Date 
# 7. Age_Category - Bins: '<18', '18-40', '40-60', '60-80', '80+' 
# 8. Wait_Category - Bins: '<30d', '30-60d', '60-90d', '90-180d', '180-365d', '>365d' 
# 9. Priority_Level - Bins: I=Critical, II=High, III=Medium, IV=Low (Acuity_Code) 
# 10. Facility_Specialty_RHA (Area) - Concatenated string combining 3 columns

class InteractiveDashboard:
    """Create interactive dashboard with filtering capabilities"""
    
    def __init__(self, df_clean):
        self.df = df_clean
        self.filtered_df = df_clean.copy()
        
    def create_filters(self):
        """Create interactive filter widgets"""
        # Status filter - default to current backlog
        status_filter = widgets.SelectMultiple(
            options=['All', 'Queued', 'Scheduled', 'Completed'],
            value=['Queued', 'Scheduled'],
            description='Status:',
            layout=widgets.Layout(width='200px')
        )
        
        # Date range by Surgery Year/Month
        year_options = [int(y) for y in self.df['Surgery_Year'].dropna().unique()]
        year_filter = widgets.SelectMultiple(
            options=['All'] + sorted(year_options),
            value=['All'],
            description='Surgery Year:',
            layout=widgets.Layout(width='200px')
        )
        
        # Facility-Specialty-RHA filter - handle NaN values
        fsr_options = self.df['Facility_Specialty_RHA'].dropna().unique().tolist()
        fsr_filter = widgets.SelectMultiple(
            options=['All'] + sorted(fsr_options)[:20],
            value=['All'],
            description='Facility-Spec-RHA:',
            layout=widgets.Layout(width='400px', height='150px')
        )

        # Specialty filter
        specialty_options = self.df['Specialty'].dropna().unique().tolist()
        specialty_filter = widgets.SelectMultiple(
            options=['All'] + sorted(specialty_options),
            value=['All'],
            description='Specialties:',
            layout=widgets.Layout(width='300px', height='150px')
        )

        # Facility filter
        facility_options = self.df['Facility'].dropna().unique().tolist()
        facility_filter = widgets.SelectMultiple(
            options=['All'] + sorted(facility_options),
            value=['All'],
            description='Facilities:',
            layout=widgets.Layout(width='300px', height='150px')
        )
        
        # Priority Level filter (using derived feature)
        priority_filter = widgets.SelectMultiple(
            options=['All', 'Critical', 'High', 'Medium', 'Low'],
            value=['All'],
            description='Priority Level:',
            layout=widgets.Layout(width='200px')
        )
        
        # Age Category filter (using derived feature)
        age_filter = widgets.SelectMultiple(
            options=['All'] + ['<18', '18-40', '40-60', '60-80', '80+'],
            value=['All'],
            description='Age Category:',
            layout=widgets.Layout(width='200px')
        )
        
        # Wait Category filter (using derived feature)
        wait_filter = widgets.SelectMultiple(
            options=['All'] + ['<30d', '30-60d', '60-90d', '90-180d', '180-365d', '>365d'],
            value=['All'],
            description='Wait Category:',
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
                
                # Status filter
                if 'All' not in status_filter.value:
                    filtered_df = filtered_df[filtered_df['Status'].isin(status_filter.value)]
                
                # Year filter
                if 'All' not in year_filter.value:
                    filtered_df = filtered_df[filtered_df['Surgery_Year'].isin(year_filter.value)]
                
                # Facility-Specialty-RHA filter
                if 'All' not in fsr_filter.value:
                    filtered_df = filtered_df[filtered_df['Facility_Specialty_RHA'].isin(fsr_filter.value)]
                
                # Priority Level filter
                if 'All' not in priority_filter.value:
                    filtered_df = filtered_df[filtered_df['Priority_Level'].isin(priority_filter.value)]
                
                # Age Category filter
                if 'All' not in age_filter.value:
                    filtered_df = filtered_df[filtered_df['Age_Category'].isin(age_filter.value)]
                
                # Wait Category filter
                if 'All' not in wait_filter.value:
                    filtered_df = filtered_df[filtered_df['Wait_Category'].isin(wait_filter.value)]
                
                self.filtered_df = filtered_df
                self.display_dashboard()
        
        update_button.on_click(update_dashboard)
        
        # Layout
        filters_box = widgets.VBox([
            widgets.HBox([status_filter, priority_filter, age_filter]),
            widgets.HBox([year_filter, wait_filter]),
            widgets.HBox([fsr_filter]),
            widgets.HBox([update_button])
        ])
        
        display(filters_box)
        display(output)
        
        # Initial display
        with output:
            self.filtered_df = self.df[self.df['Status'].isin(['Queued', 'Scheduled'])]
            self.display_dashboard()
    
    def display_dashboard(self):
        """Display the main dashboard"""
        # Calculate metrics
        active_waitlist = self.filtered_df[self.filtered_df['Status'].isin(['Queued', 'Scheduled'])]
        current_backlog = active_waitlist['Patient_Id'].nunique()
        
        # Current wait from Recommend_Surgery_By_Date
        active_waitlist['Current_Wait_Days'] = (pd.Timestamp.now() - active_waitlist['Recommend_Surgery_By_Date']).dt.days
        
        # Create figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Wait Trend by Month', 'Backlog by Priority Level',
                'Wait Category Distribution', 'Backlog Metrics',
                'Age Category vs Wait Time', 'Top 10 Facility-Specialty-RHA'
            ),
            specs=[[{'secondary_y': True}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'indicator'}],
                   [{'type': 'box'}, {'type': 'bar'}]],
            row_heights=[0.3, 0.3, 0.4]
        )
        
        # 1. Monthly trend using Surgery_Month
        if len(active_waitlist) > 0:
            monthly_backlog = active_waitlist.groupby('Surgery_Month').agg({
                'Current_Wait_Days': ['mean', 'count']
            })
            
            fig.add_trace(
                go.Scatter(x=monthly_backlog.index,
                          y=monthly_backlog['Current_Wait_Days']['mean'],
                          name='Avg Wait',
                          line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=monthly_backlog.index,
                       y=monthly_backlog['Current_Wait_Days']['count'],
                       name='Count',
                       opacity=0.3),
                row=1, col=1,
                secondary_y=True
            )
        
        # 2. Backlog by Priority Level
        priority_counts = active_waitlist['Priority_Level'].value_counts()
        fig.add_trace(
            go.Bar(x=priority_counts.index,
                   y=priority_counts.values,
                   marker_color=['red', 'orange', 'yellow', 'green']),
            row=1, col=2
        )
        
        # 3. Wait Category Distribution
        wait_cat_counts = active_waitlist['Wait_Category'].value_counts()
        fig.add_trace(
            go.Bar(x=wait_cat_counts.index,
                   y=wait_cat_counts.values,
                   marker_color='lightblue'),
            row=2, col=1
        )
        
        # 4. Backlog Gauge
        avg_wait = np.ceil(active_waitlist['Current_Wait_Days'].mean()) if len(active_waitlist) > 0 else 0
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=current_backlog,
                title={"text": f"Current Backlog<br>Avg Wait: {avg_wait} days"},
                gauge={'axis': {'range': [0, 20000]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 5000], 'color': "lightgreen"},
                           {'range': [5000, 10000], 'color': "yellow"},
                           {'range': [10000, 20000], 'color': "lightcoral"}]},
                number={'font': {'size': 40}}
            ),
            row=2, col=2
        )
        
        # 5. Age Category Analysis
        age_waits = active_waitlist.groupby('Age_Category')['Current_Wait_Days'].apply(list)
        for age_cat in ['<18', '18-40', '40-60', '60-80', '80+']:
            if age_cat in age_waits.index:
                fig.add_trace(
                    go.Box(y=age_waits[age_cat], name=age_cat),
                    row=3, col=1
                )
        
        # 6. Top Facility-Specialty-RHA combinations
        fsr_counts = active_waitlist['Facility_Specialty_RHA'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=fsr_counts.values,
                   y=fsr_counts.index,
                   orientation='h',
                   marker_color='orange'),
            row=3, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Month", row=1, col=1)
        fig.update_yaxes(title_text="Avg Wait (days)", row=1, col=1)
        fig.update_yaxes(title_text="Patient Count", row=1, col=1, secondary_y=True)
        
        fig.update_xaxes(title_text="Priority Level", row=1, col=2)
        fig.update_yaxes(title_text="Number of Patients", row=1, col=2)
        
        fig.update_xaxes(title_text="Wait Category", row=2, col=1)
        fig.update_yaxes(title_text="Number of Patients", row=2, col=1)
        
        fig.update_xaxes(title_text="Age Category", row=3, col=1)
        fig.update_yaxes(title_text="Current Wait (days)", row=3, col=1)
        
        fig.update_xaxes(title_text="Number of Patients", row=3, col=2)
        
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text=f"Healthcare Waitlist Analytics | Backlog: {current_backlog:,} patients | Filters Applied: {len(self.filtered_df):,} records"
        )
        
        fig.show()
        
        # Summary statistics by derived features
        print("\nBACKLOG ANALYSIS BY DERIVED FEATURES")
        print("="*50)
        print(f"Total Backlog: {current_backlog:,} patients")
        
        if len(active_waitlist) > 0:
            print(f"\nBy Priority Level:")
            for level in ['Critical', 'High', 'Medium', 'Low']:
                count = (active_waitlist['Priority_Level'] == level).sum()
                pct = count/len(active_waitlist)*100
                print(f"  {level}: {count:,} ({pct:.1f}%)")
            
            print(f"\nBy Age Category:")
            for cat in active_waitlist['Age_Category'].value_counts().index[:5]:
                count = (active_waitlist['Age_Category'] == cat).sum()
                print(f"  {cat}: {count:,}")
            
            print(f"\nBy Wait Category:")
            for cat in active_waitlist['Wait_Category'].value_counts().index:
                count = (active_waitlist['Wait_Category'] == cat).sum()
                print(f"  {cat}: {count:,}")

# Create and display interactive dashboard
dashboard = InteractiveDashboard(df_clean)
dashboard.create_filters()