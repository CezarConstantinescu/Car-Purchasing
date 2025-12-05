import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Optional interactive plotting
try:
    import plotly.express as px
except Exception:
    px = None

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

    # Map controls and interactive map (standalone above the insights columns)
    st.subheader("📍 Sales by State (US map)")
    if px is None:
        st.warning("Plotly is not installed. Install it with `pip install plotly` to see the interactive US map.")
    else:
        with st.expander("Map controls", expanded=False):
            map_metric = st.selectbox("Metric to display on map", ["Total Sales", "Average Sale Price", "Number of Sales"], index=0)
            # Sale Month filter for the map
            month_col = 'Sale Month' if 'Sale Month' in filtered_df.columns else None
            month_options = sorted(filtered_df[month_col].dropna().unique().tolist()) if month_col else []
            selected_months = st.multiselect("Filter months (map)", options=month_options, default=month_options)
            color_scale = st.selectbox("Color scale", ["Blues", "Viridis", "Reds", "Cividis"], index=0)

        # Mapping of full state names to USPS abbreviations
        state_abbrev = {
            'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
        }

        # Build the DataFrame used for the map, applying the optional month filter
        map_df = filtered_df.copy()
        if month_col and selected_months:
            map_df = map_df[map_df[month_col].isin(selected_months)]

        # Aggregate by region/state depending on chosen metric
        if map_metric == "Total Sales":
            agg = map_df.groupby(region_col)[sale_col].sum().reset_index().rename(columns={region_col: 'state_name', sale_col: 'value'})
            value_label = 'Total Sales ($)'
        elif map_metric == "Average Sale Price":
            agg = map_df.groupby(region_col)[sale_col].mean().reset_index().rename(columns={region_col: 'state_name', sale_col: 'value'})
            value_label = 'Avg Sale Price ($)'
        else:  # Number of Sales
            agg = map_df.groupby(region_col).size().reset_index(name='value').rename(columns={region_col: 'state_name'})
            value_label = 'Number of Sales'

        # Map to state codes and prepare hover info
        agg['state_code'] = agg['state_name'].map(state_abbrev)
        # add additional hover metrics
        stats = map_df.groupby(region_col)[sale_col].agg(['sum', 'mean', 'count']).reset_index().rename(columns={region_col: 'state_name'})
        merged = agg.merge(stats, on='state_name', how='left')
        merged = merged.dropna(subset=['state_code'])

        if merged.empty:
            st.info("No valid US state names found in the Sales Region column (after filters) to render the map.")
        else:
            # Prepare user-friendly hover fields (no = or underscores)
            merged['sum_fmt'] = merged['sum'].apply(lambda x: f"${x:,.2f}")
            merged['mean_fmt'] = merged['mean'].apply(lambda x: f"${x:,.2f}")
            merged['count'] = merged['count'].astype(int)

            labels_map = {
                'value': value_label,
                'state_name': 'State',
                'sum_fmt': 'Total sales',
                'mean_fmt': 'Average sale',
                'count': 'Number of sales'
            }

            # Use custom_data + hovertemplate for exact hover formatting
            fig_map = px.choropleth(
                merged,
                locations='state_code',
                locationmode='USA-states',
                color='value',
                scope='usa',
                color_continuous_scale=color_scale,
                labels={'value': value_label},
                hover_name='state_name',
                custom_data=['sum_fmt', 'mean_fmt', 'count']
            )

            hovertemplate = (
                "<b>%{hovertext}</b><br>"
                "Total sales: %{customdata[0]}<br>"
                "Average sale: %{customdata[1]}<br>"
                "Number of sales: %{customdata[2]}<extra></extra>"
            )

            fig_map.update_traces(marker_line_width=0.5, marker_line_color='white', hovertemplate=hovertemplate)
            fig_map.update_layout(margin={'r':0,'t':30,'l':0,'b':0}, height=480)
            st.plotly_chart(fig_map, use_container_width=True)

            # State inspector: select a state to show details
            state_options = merged['state_name'].sort_values().tolist()
            if state_options:
                sel_state = st.selectbox('Inspect state', options=state_options, index=0)
                st.markdown(f"**Summary for {sel_state}**")
                s = merged[merged['state_name'] == sel_state].iloc[0]
                st.metric('Total sales', f"${s['sum']:,.2f}")
                st.metric('Average sale', f"${s['mean']:,.2f}")
                st.metric('Number of sales', int(s['count']))

                # Top car makes in that state
                make_col = 'Car Make' if 'Car Make' in map_df.columns else None
                if make_col:
                    top_makes = (map_df[map_df[region_col] == sel_state].groupby(make_col)[sale_col].sum().reset_index().sort_values(by=sale_col, ascending=False).head(5))
                    top_makes.columns = ['Car Make', 'Total Sales']
                    st.table(top_makes)
                else:
                    st.info('No `Car Make` column found to show top makes.')
        # (removed legacy non-interactive map block)

    # Statistical summary
    st.header("📊 Statistical Summary")
    st.dataframe(
        filtered_df.describe(),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("*Business Intelligence Dashboard for Car Purchasing Decisions*")
