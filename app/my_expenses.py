"""
app.py
Streamlit application for visualizing and predicting student expenses
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Student Expenses Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)
@st.cache_data
def load_data():
    """Load expense data from CSV"""
    path = "cedric_yearly_expenses_2024.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.error(f"CSV file not found at {path}")
        return pd.DataFrame()  # return empty DataFrame    

    # Add additional columns for analysis
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    df['Quarter'] = df['Date'].dt.quarter
    df['Season'] = df['Date'].apply(lambda x: get_season(x.month))
    
    return df

def get_season(month):
    """Determines season from month"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


def train_prediction_model(df):
    """Train a linear regression model to predict monthly expenses"""
    # Aggregate data by month
    monthly_data = df.groupby(['Month', 'Season']).agg({
        'Amount': 'sum',
        'Internship_period': 'first',
        'Part_time_job_period': 'first'
    }).reset_index()
    
    # Prepare features
    X = monthly_data[['Month']].copy()
    X['Internship'] = monthly_data['Internship_period'].astype(int)
    X['Part_time'] = monthly_data['Part_time_job_period'].astype(int)
    
    # Encode season
    le = LabelEncoder()
    X['Season_encoded'] = le.fit_transform(monthly_data['Season'])
    
    # Target variable
    y = monthly_data['Amount']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    return model, le


def main():
    """Main application function"""
    
    # Title and description
    st.title("📊 Student Expenses Tracker & Predictor")
    st.markdown("### Visualize your expenses and predict future spending patterns")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Category filter
    categories = st.sidebar.multiselect(
        "Select Categories",
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Date'].min(), df['Date'].max()),
        min_value=df['Date'].min(),
        max_value=df['Date'].max()
    )
    
    # Apply filters
    if len(date_range) == 2:
        filtered_df = df[
            (df['Category'].isin(categories)) &
            (df['Date'] >= pd.to_datetime(date_range[0])) &
            (df['Date'] <= pd.to_datetime(date_range[1]))
        ]
    else:
        filtered_df = df[df['Category'].isin(categories)]
    
    # KPI Metrics
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_expenses = filtered_df['Amount'].sum()
        st.metric("Total Expenses", f"€{total_expenses:,.2f}")
    
    with col2:
        top_category = filtered_df.groupby('Category')['Amount'].sum().idxmax() if not filtered_df.empty else "N/A"
        top_amount = filtered_df[filtered_df['Category'] == top_category]['Amount'].sum() if top_category != "N/A" else 0
        st.metric("Top Category", top_category, f"€{top_amount:,.2f}")
    
    with col3:
        avg_monthly = filtered_df.groupby('Month')['Amount'].sum().mean() if not filtered_df.empty else 0
        st.metric("Avg Monthly Expenses", f"€{avg_monthly:,.2f}")
    
    with col4:
        num_transactions = len(filtered_df)
        st.metric("Total Transactions", f"{num_transactions:,}")
    
    st.markdown("---")
    
    # Charts Section
    st.markdown("### 📊 Expense Visualizations")
    
    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Pie chart of expenses by category
        if not filtered_df.empty:
            category_sum = filtered_df.groupby('Category')['Amount'].sum().reset_index()
            fig_pie = px.pie(
                category_sum, 
                values='Amount', 
                names='Category',
                title='Expenses by Category',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    with chart_col2:
        # Bar chart of expenses per month
        if not filtered_df.empty:
            monthly_expenses = filtered_df.groupby('Month_Name')['Amount'].sum().reset_index()
            # Sort by month order
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_expenses['Month_Name'] = pd.Categorical(monthly_expenses['Month_Name'], 
                                                           categories=month_order, 
                                                           ordered=True)
            monthly_expenses = monthly_expenses.sort_values('Month_Name')
            
            fig_bar = px.bar(
                monthly_expenses,
                x='Month_Name',
                y='Amount',
                title='Monthly Expenses',
                color='Amount',
                color_continuous_scale='Viridis'
            )
            # fig_bar.update_xaxis(tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    # Line chart of cumulative expenses
    if not filtered_df.empty:
        st.markdown("### 📈 Cumulative Expenses Over Time")
        cumulative_df = filtered_df.sort_values('Date').copy()
        cumulative_df['Cumulative_Amount'] = cumulative_df['Amount'].cumsum()
        
        fig_line = px.line(
            cumulative_df,
            x='Date',
            y='Cumulative_Amount',
            title='Cumulative Expenses Throughout the Year',
            color_discrete_sequence=['#FF6B6B']
        )
        fig_line.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative Amount (€)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("---")
    
    # Prediction Section
    st.markdown("### 🔮 Expense Prediction")
    st.markdown("Predict your monthly expenses based on historical patterns")
    
    # Train model
    model, season_encoder = train_prediction_model(df)
    
    # User input for prediction
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        selected_month = st.selectbox(
            "Select Month",
            options=list(range(1, 13)),
            format_func=lambda x: datetime(2024, x, 1).strftime('%B')
        )
    
    with pred_col2:
        internship_flag = st.selectbox(
            "Internship Period?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
    
    with pred_col3:
        part_time_flag = st.selectbox(
            "Part-time Job Period?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
    
    # Make prediction
    if st.button("🎯 Predict Monthly Expenses", type="primary"):
        # Prepare input
        season = get_season(selected_month)
        season_encoded = season_encoder.transform([season])[0]
        
        X_pred = np.array([[selected_month, internship_flag, part_time_flag, season_encoded]])
        prediction = model.predict(X_pred)[0]
        
        # Display prediction
        st.success(f"### Predicted Monthly Expenses: €{prediction:,.2f}")
        
        # Show comparison with actual average
        actual_avg = df[df['Month'] == selected_month]['Amount'].sum()
        diff = prediction - actual_avg
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Amount", f"€{prediction:,.2f}")
        with col2:
            st.metric("Actual (2024)", f"€{actual_avg:,.2f}", delta=f"€{diff:,.2f}")
        
        # Additional insights
        st.info(f"""
        **Insights:**
        - Month: {datetime(2024, selected_month, 1).strftime('%B')}
        - Season: {season}
        - Internship: {'Yes' if internship_flag else 'No'}
        - Part-time Job: {'Yes' if part_time_flag else 'No'}
        
        This prediction is based on your historical spending patterns in similar conditions.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
            <p>Student Expense Tracker v1.0 | Built with Streamlit 💙</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()