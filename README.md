# 📊 Student Expenses Tracker & Predictor

A comprehensive Streamlit application for visualizing and predicting my expenses as a student throughout the year. This app generates synthetic expense data and provides insightful visualizations along with machine learning-based expense predictions.

## ✨ Features

- **📈 Interactive Dashboard**: Real-time visualization of expenses with filters
- **💰 Expense Categories**: Track spending across multiple categories: Groceries shoping
- **📊 Multiple Visualizations**: 
  - Pie chart for category distribution
  - Bar chart for monthly expenses
  - Line chart for cumulative spending
- **🔮 ML Predictions**: Predict monthly expenses using Linear Regression (will be using other regression models potentially)
- **🎯 Smart Filters**: Filter by category and date range

## 🏗️ Project Structure

```
.
├── app/                                 # Streamlit app and data
│   ├── my_expenses.py                   # Main Streamlit application
│   ├── requirements.txt                 # Python dependencies
│   ├── Dockerfile                       # Docker configuration for containerization
│   └── cedric_yearly_expenses_2024.csv  # Synthetic expense data (generated)
├── .github/                             # GitHub Actions workflows for CI/CD
│   └── workflows/
│       └── ci.yml                       # Continuous Integration and Deployment pipeline
├── images/                              # Screenshots for the README
│   ├── pie_chart.png
│   ├── bar_chart.png
├── test_my_expenses.py                  # Comprehensive test suite (Unit, Integration, Regression, Performance)
└── README.md                            # Project documentation

```


## 📋 Prerequisites

- pip (Python package manager)
- Docker (for containerized deployment)



## 🚀 Quick Start

### Option 1: Run with Docker
1. **Build the Docker image:**
[Bash]
docker build -t student-expenses-app ./app

2. **Run the Docker container**
[Bash]
docker run -p 8501:8501 student-expenses-app

3. **Access the application**
Open the desktop app Docker to launch the website or go to http://localhost:8501.


### Option 2: Run Locally

1. **Clone the repository:**
[Bash]
git clone https://github.com/cedrickirek/my_past_and_future_expenses.git
cd expenses-app

2. **Navigate to the app directory and install dependencies:**
[Bash]
cd app
pip install -r requirements.txt

3. **Run the Streamlit app:**
streamlit run my_expenses.py

4. **Open your browser** and go to `http://localhost:8501`




## 🚀 Testing and CI/CD

This project is equipped with a robust CI/CD pipeline, defined in .github/workflows/ci.yml.

1. **Unit Tests:** Validate individual functions like data loading and seasonal mapping.

2. **Integration Tests:** Ensure different components, such as the machine learning model and data processing, work together seamlessly.

3. **Performance Tests:** Analyze the application's performance with large datasets.

4. **Docker Tests:** Build and test the Docker container to ensure successful deployment.

*All tests are automatically executed on every push and pull request to the repository, ensuring code quality and stability.*




## 🎯 Using the Prediction Feature

1. Select a month from the dropdown
2. Choose whether it's an internship period (Yes/No)
3. Choose whether it's a part-time job period (Yes/No)
4. Click "Predict Monthly Expenses"
5. View the predicted amount and comparison with historical data

## 🛠️ Customization

### Updating the UI
Edit `my_expenses.py` to modify:
- Chart types and colors
- Layout and styling
- KPI metrics
- Prediction model features


## 🐛 Troubleshooting

### Common Issues:

1. **Port already in use:**
   - Change the port: `streamlit run app.py --server.port=8502`

2. **Docker build fails:**
   - Ensure Docker daemon is running
   - Check internet connection for package downloads

## 🤝 Contributing and License

This project is private and intended for a class assignment. Nevertheless, Feel free to enhance this project by:

- suggesting different categorisations
- Creating additional visualizations
- Adding data export features
- Implementing budget tracking

**License:** This project is for private use.
