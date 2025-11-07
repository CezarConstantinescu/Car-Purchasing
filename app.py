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

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('car_data.csv')
        return df
    except FileNotFoundError:
        st.error("Error: car_data.csv file not found. Please ensure the data file exists in the same directory as the application.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Gender filter
gender_options = ['All'] + list(df['Gender'].unique())
selected_gender = st.sidebar.selectbox("Select Gender", gender_options)

# Age filter
min_age = int(df['Age'].min())
max_age = int(df['Age'].max())
age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

# Country filter
country_options = ['All'] + sorted(df['Country'].unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", country_options)

# Annual Salary filter
min_salary = float(df['Annual Salary'].min())
max_salary = float(df['Annual Salary'].max())
salary_range = st.sidebar.slider("Select Annual Salary Range", min_salary, max_salary, (min_salary, max_salary))

# Apply filters
filtered_df = df.copy()

if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]

filtered_df = filtered_df[
    (filtered_df['Age'] >= age_range[0]) & 
    (filtered_df['Age'] <= age_range[1])
]

if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]

filtered_df = filtered_df[
    (filtered_df['Annual Salary'] >= salary_range[0]) & 
    (filtered_df['Annual Salary'] <= salary_range[1])
]

# Key metrics
st.header("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(filtered_df))

with col2:
    if len(filtered_df) > 0:
        avg_purchase = filtered_df['Car Purchase Amount'].mean()
        st.metric("Avg Car Purchase", f"${avg_purchase:,.2f}")
    else:
        st.metric("Avg Car Purchase", "N/A")

with col3:
    if len(filtered_df) > 0:
        avg_salary = filtered_df['Annual Salary'].mean()
        st.metric("Avg Annual Salary", f"${avg_salary:,.2f}")
    else:
        st.metric("Avg Annual Salary", "N/A")

with col4:
    if len(filtered_df) > 0:
        avg_net_worth = filtered_df['Net Worth'].mean()
        st.metric("Avg Net Worth", f"${avg_net_worth:,.2f}")
    else:
        st.metric("Avg Net Worth", "N/A")

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
        file_name='filtered_car_data.csv',
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
    # Car Purchase Amount Distribution
    st.subheader("Car Purchase Amount Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(filtered_df['Car Purchase Amount'], bins=20, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Car Purchase Amount ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Car Purchase Amounts')
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close()

with chart_col2:
    # Gender Distribution
    st.subheader("Gender Distribution")
    gender_counts = filtered_df['Gender'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = ['#ff9999', '#66b3ff']
    ax2.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=colors)
    ax2.set_title('Customer Gender Distribution')
    st.pyplot(fig2)
    plt.close()

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    # Age vs Car Purchase Amount
    st.subheader("Age vs Car Purchase Amount")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(filtered_df['Age'], filtered_df['Car Purchase Amount'], 
                alpha=0.6, c='green', edgecolors='black')
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Car Purchase Amount ($)')
    ax3.set_title('Age vs Car Purchase Amount')
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close()

with chart_col4:
    # Annual Salary vs Car Purchase Amount
    st.subheader("Annual Salary vs Car Purchase Amount")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.scatter(filtered_df['Annual Salary'], filtered_df['Car Purchase Amount'], 
                alpha=0.6, c='orange', edgecolors='black')
    ax4.set_xlabel('Annual Salary ($)')
    ax4.set_ylabel('Car Purchase Amount ($)')
    ax4.set_title('Annual Salary vs Car Purchase Amount')
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)
    plt.close()

    # Additional analysis
    st.header("🔍 Additional Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        # Top 10 countries by average purchase
        st.subheader("Top 10 Countries by Avg Purchase Amount")
        country_avg = filtered_df.groupby('Country')['Car Purchase Amount'].mean().sort_values(ascending=False).head(10)
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        country_avg.plot(kind='barh', ax=ax5, color='coral')
        ax5.set_xlabel('Average Car Purchase Amount ($)')
        ax5.set_ylabel('Country')
        ax5.set_title('Top 10 Countries by Average Purchase Amount')
        ax5.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig5)
        plt.close()

    with insight_col2:
        # Age group analysis
        st.subheader("Age Group Analysis")
        age_bins = [0, 30, 40, 50, 60, 100]
        age_labels = ['<30', '30-40', '40-50', '50-60', '60+']
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['Age Group'] = pd.cut(filtered_df_copy['Age'], bins=age_bins, labels=age_labels)
        age_group_avg = filtered_df_copy.groupby('Age Group')['Car Purchase Amount'].mean()
        
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        age_group_avg.plot(kind='bar', ax=ax6, color='lightgreen', edgecolor='black')
        ax6.set_xlabel('Age Group')
        ax6.set_ylabel('Average Car Purchase Amount ($)')
        ax6.set_title('Average Purchase Amount by Age Group')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()

    # Net Worth vs Purchase Analysis
    st.subheader("Net Worth vs Car Purchase Amount")
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    scatter = ax7.scatter(filtered_df['Net Worth'], filtered_df['Car Purchase Amount'], 
                         c=filtered_df['Age'], cmap='viridis', alpha=0.6, edgecolors='black')
    ax7.set_xlabel('Net Worth ($)')
    ax7.set_ylabel('Car Purchase Amount ($)')
    ax7.set_title('Net Worth vs Car Purchase Amount (colored by Age)')
    ax7.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label('Age')
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
