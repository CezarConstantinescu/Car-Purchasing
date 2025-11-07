# Car-Purchasing
Business Intelligence application for supporting the decision making for the process of purchasing a car.

## Overview
This application provides an interactive web-based dashboard for analyzing car purchasing data. It uses Python with pandas for data manipulation, matplotlib for visualizations, and Streamlit for the web interface.

## Features
- 📊 Interactive data grids with filtering capabilities
- 📈 Multiple chart types including histograms, scatter plots, pie charts, and bar charts
- 🔍 Advanced filtering by gender, age, country, and salary range
- 💡 Key metrics and statistical summaries
- 📥 Export filtered data to CSV
- 🎨 Multiple visualizations for data exploration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CezarConstantinescu/Car-Purchasing.git
cd Car-Purchasing
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Data Structure

The application uses a CSV file (`car_data.csv`) with the following columns:
- Customer Name
- Customer E-mail
- Country
- Gender
- Age
- Annual Salary
- Credit Card Debt
- Net Worth
- Car Purchase Amount

## Dashboard Features

### Filters
Use the sidebar to filter data by:
- Gender
- Age Range
- Country
- Annual Salary Range

### Key Metrics
View important metrics including:
- Total number of customers
- Average car purchase amount
- Average annual salary
- Average net worth

### Visualizations
- **Car Purchase Amount Distribution**: Histogram showing the distribution of purchase amounts
- **Gender Distribution**: Pie chart showing customer gender breakdown
- **Age vs Car Purchase Amount**: Scatter plot showing relationship between age and purchase amount
- **Annual Salary vs Car Purchase Amount**: Scatter plot showing correlation between salary and purchase
- **Top 10 Countries**: Bar chart of countries by average purchase amount
- **Age Group Analysis**: Bar chart showing average purchase by age group
- **Net Worth vs Purchase Amount**: Scatter plot colored by age

### Data Export
Download filtered data as CSV for further analysis.

## Technologies Used
- Python 3.x
- pandas - Data manipulation and analysis
- matplotlib - Data visualization
- streamlit - Web interface
- numpy - Numerical computing

## License
This project is open source and available under the MIT License.
