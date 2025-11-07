import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Car Purchasing BI Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 Car Purchasing Business Intelligence Dashboard")
st.markdown("### Supporting Decision Making for Car Purchases")

# Load data (adapted to car_sales.csv schema)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('car_sales.csv')
        return df
    except FileNotFoundError:
        st.error("Error: car_sales.csv file not found. Please ensure the data file exists in the same directory as the application.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Gender filter (file uses 'Customer Gender')
gender_col = 'Customer Gender' if 'Customer Gender' in df.columns else 'Gender'
gender_options = ['All'] + list(df[gender_col].dropna().unique())
selected_gender = st.sidebar.selectbox("Select Gender", gender_options)

# Age filter (file uses 'Customer Age')
age_col = 'Customer Age' if 'Customer Age' in df.columns else 'Age'
min_age = int(df[age_col].min())
max_age = int(df[age_col].max())
age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

# Sales region filter (replaces Country)
region_col = 'Sales Region' if 'Sales Region' in df.columns else 'Country'
region_options = ['All'] + sorted(df[region_col].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Select Sales Region", region_options)

# Income filter (replaces Annual Salary)
income_col = 'Income' if 'Income' in df.columns else 'Annual Salary'
min_income = float(df[income_col].min())
max_income = float(df[income_col].max())
income_range = st.sidebar.slider("Select Income Range", min_income, max_income, (min_income, max_income))

# Apply filters
filtered_df = df.copy()

if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df[gender_col] == selected_gender]

filtered_df = filtered_df[
    (filtered_df[age_col] >= age_range[0]) & 
    (filtered_df[age_col] <= age_range[1])
]

if selected_region != 'All':
    filtered_df = filtered_df[filtered_df[region_col] == selected_region]

filtered_df = filtered_df[
    (filtered_df[income_col] >= income_range[0]) & 
    (filtered_df[income_col] <= income_range[1])
]

# Key metrics
st.header("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(filtered_df))

with col2:
    if len(filtered_df) > 0:
        # Sale amount in car_sales.csv is 'Sale Price'
        sale_col = 'Sale Price' if 'Sale Price' in filtered_df.columns else 'Car Purchase Amount'
        avg_purchase = filtered_df[sale_col].mean()
        st.metric("Avg Sale Price", f"${avg_purchase:,.2f}")
    else:
        st.metric("Avg Sale Price", "N/A")

with col3:
    if len(filtered_df) > 0:
        avg_income = filtered_df[income_col].mean()
        st.metric("Avg Income", f"${avg_income:,.2f}")
    else:
        st.metric("Avg Income", "N/A")

with col4:
    if len(filtered_df) > 0:
        # dataset doesn't include Net Worth; show average car year instead
        car_year_col = 'Car Year' if 'Car Year' in filtered_df.columns else 'Year'
        try:
            avg_car_year = int(filtered_df[car_year_col].mean())
            st.metric("Avg Car Year", f"{avg_car_year}")
        except Exception:
            st.metric("Avg Car Year", "N/A")
    else:
        st.metric("Avg Car Year", "N/A")

# Data grid
st.header("📋 Customer Data")

if len(filtered_df) == 0:
    st.warning("No data matches the selected filters. Please adjust your filter criteria.")
else:
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_car_sales.csv',
        mime='text/csv',
    )

# Charts section
st.header("📈 Data Visualizations")

if len(filtered_df) == 0:
    st.info("No data available for visualization. Please adjust your filter criteria.")
else:
    # Two columns for charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Sale Price Distribution
        st.subheader("Sale Price Distribution")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.hist(filtered_df[sale_col], bins=20, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Sale Price ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Sale Prices')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close()

    with chart_col2:
        # Gender Distribution
        st.subheader("Gender Distribution")
        gender_counts = filtered_df[gender_col].value_counts()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax2.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=colors[:len(gender_counts)])
        ax2.set_title('Customer Gender Distribution')
        st.pyplot(fig2)
        plt.close()

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # Age vs Sale Price
        st.subheader("Age vs Sale Price")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(filtered_df[age_col], filtered_df[sale_col], 
                    alpha=0.6, c='green', edgecolors='black')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Sale Price ($)')
        ax3.set_title('Age vs Sale Price')
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close()

    with chart_col4:
        # Income vs Sale Price
        st.subheader("Income vs Sale Price")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.scatter(filtered_df[income_col], filtered_df[sale_col], 
                    alpha=0.6, c='orange', edgecolors='black')
        ax4.set_xlabel('Income ($)')
        ax4.set_ylabel('Sale Price ($)')
        ax4.set_title('Income vs Sale Price')
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close()

    # Additional analysis
    st.header("🔍 Additional Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        # Top 10 sales regions by average sale price
        st.subheader("Top 10 Sales Regions by Avg Sale Price")
        region_avg = filtered_df.groupby(region_col)[sale_col].mean().sort_values(ascending=False).head(10)
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        region_avg.plot(kind='barh', ax=ax5, color='coral')
        ax5.set_xlabel('Average Sale Price ($)')
        ax5.set_ylabel('Sales Region')
        ax5.set_title('Top 10 Sales Regions by Average Sale Price')
        ax5.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig5)
        plt.close()

    with insight_col2:
        # Age group analysis
        st.subheader("Age Group Analysis")
        age_bins = [0, 30, 40, 50, 60, 100]
        age_labels = ['<30', '30-40', '40-50', '50-60', '60+']
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['Age Group'] = pd.cut(filtered_df_copy[age_col], bins=age_bins, labels=age_labels)
        age_group_avg = filtered_df_copy.groupby('Age Group')[sale_col].mean()
        
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        age_group_avg.plot(kind='bar', ax=ax6, color='lightgreen', edgecolor='black')
        ax6.set_xlabel('Age Group')
        ax6.set_ylabel('Average Sale Price ($)')
        ax6.set_title('Average Sale Price by Age Group')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()

    # Car Year vs Sale Price
    st.subheader("Car Year vs Sale Price")
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    # color by income to surface purchasing power
    scatter = ax7.scatter(filtered_df[car_year_col], filtered_df[sale_col], 
                         c=filtered_df[income_col], cmap='viridis', alpha=0.6, edgecolors='black')
    ax7.set_xlabel('Car Year')
    ax7.set_ylabel('Sale Price ($)')
    ax7.set_title('Car Year vs Sale Price (colored by Income)')
    ax7.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label('Income')
    st.pyplot(fig7)
    plt.close()

    # Statistical summary
    st.header("📊 Statistical Summary")
    st.dataframe(
        filtered_df.describe(),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("*Business Intelligence Dashboard for Car Purchasing Decisions*")
