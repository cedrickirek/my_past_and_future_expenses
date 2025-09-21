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
/app
  ├── my_expenses.py                      # Main Streamlit application
  ├── requirements.txt                    # Python dependencies
  ├── Dockerfile                          # Docker configuration
  ├── cedric_yearly_expenses_2024.csv     # csv file with expenses
  ├── .gitignore                          # files to be ignored 
  └── README.md                           # Project documentation
/.github
/test_my_expenses.py

```

## 📋 Prerequisites

- pip (Python package manager)
- Docker (optional, for containerized deployment)

## 🚀 Quick Start

### Option 1: Run Locally

1. **Clone or create the project directory:**
```bash
mkdir expenses-app
cd expenses-app
```

2. **Create all the project files** (copy the provided code into respective files)

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Generate the data:**
```bash
python generate_data.py
```

5. **Run the Streamlit app:**
```bash
streamlit run app.py
```

6. **Open your browser** and navigate to `http://localhost:8501`

### Option 2: Run with Docker

1. **Build the Docker image:**
```bash
docker build -t student-expenses-app .
```

2. **Run the Docker container:**
```bash
docker run -p 8501:8501 student-expenses-app
```

3. **Access the app** at `http://localhost:8501`


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

## 📦 Dependencies

- **streamlit**: Web app framework
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning models

## 🐛 Troubleshooting

### Common Issues:

1. **Port already in use:**
   - Change the port: `streamlit run app.py --server.port=8502`

2. **Data not generating:**
   - Ensure `generate_data.py` runs successfully
   - Check for `student_expenses_2024.csv` in the project directory

3. **Docker build fails:**
   - Ensure Docker daemon is running
   - Check internet connection for package downloads

## 🤝 Contributing

Feel free to enhance this project by:
- suggesting different categorisations
- Creating additional visualizations
- Adding data export features
- Implementing budget tracking

## 📄 License

This project is private 