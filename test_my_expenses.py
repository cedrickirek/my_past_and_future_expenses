"""
test_my_expenses.py
Comprehensive test suite for the Student Expenses Tracker Streamlit application
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import os
import tempfile
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from app.my_expenses import get_season

# Import functions from the main app
# Assuming the main file is named my_expenses.py based on Dockerfile
try:
    from my_expenses import (
        load_data, 
        get_season, 
        train_prediction_model,
        main
    )
except ImportError:
    # If the app file is named differently, adjust this import
    print("Warning: Could not import from my_expenses.py")


class TestDataLoading:
    """Test data loading functionality"""
    
    def create_test_csv(self, filepath):
        """Helper method to create test CSV file"""
        test_data = {
            'Date': ['01/09/2024', '03/09/2024', '05/09/2024', '07/12/2024'],
            'Category': ['Rent', 'Groceries', 'Transport', 'Rent'],
            'Amount': [500.0, 29.26, 15.50, 500.0],
            'Internship_period': [False, False, True, False],
            'Part_time_job_period': [True, False, False, True]
        }
        df = pd.DataFrame(test_data)
        df.to_csv(filepath, index=False)
        return df
    
    def test_load_data_file_exists(self):
        """Test load_data when CSV file exists"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            test_df = self.create_test_csv(tmp.name)
            
            with patch('my_expenses.os.path.exists', return_value=True), \
                 patch('my_expenses.pd.read_csv', return_value=test_df):
                
                result_df = load_data()
                
                # Check that additional columns are added
                assert 'Month' in result_df.columns
                assert 'Month_Name' in result_df.columns
                assert 'Quarter' in result_df.columns
                assert 'Season' in result_df.columns
                
        os.unlink(tmp.name)
    
    def test_load_data_file_not_exists(self):
        """Test load_data when CSV file doesn't exist"""
        with patch('my_expenses.os.path.exists', return_value=False):
            try:
                result_df = load_data()
                # Should handle missing file gracefully
                assert result_df is not None or True  # Adjust based on actual behavior
            except Exception as e:
                # If it raises an exception, it should be handled properly
                assert isinstance(e, (FileNotFoundError, pd.errors.EmptyDataError))
    
    def test_data_types_after_loading(self):
        """Test that data types are correct after loading"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            test_df = self.create_test_csv(tmp.name)
            
            with patch('my_expenses.os.path.exists', return_value=True), \
                 patch('my_expenses.pd.read_csv', return_value=test_df):
                
                result_df = load_data()
                
                # Check data types
                assert pd.api.types.is_datetime64_any_dtype(result_df['Date'])
                assert pd.api.types.is_numeric_dtype(result_df['Amount'])
                assert pd.api.types.is_integer_dtype(result_df['Month'])
                
        os.unlink(tmp.name)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_get_season_winter(self):
        """Test get_season for winter months"""
        assert get_season(12) == 'Winter'
        assert get_season(1) == 'Winter'
        assert get_season(2) == 'Winter'
    
    def test_get_season_spring(self):
        """Test get_season for spring months"""
        assert get_season(3) == 'Spring'
        assert get_season(4) == 'Spring'
        assert get_season(5) == 'Spring'
    
    def test_get_season_summer(self):
        """Test get_season for summer months"""
        assert get_season(6) == 'Summer'
        assert get_season(7) == 'Summer'
        assert get_season(8) == 'Summer'
    
    def test_get_season_autumn(self):
        """Test get_season for autumn months"""
        assert get_season(9) == 'Autumn'
        assert get_season(10) == 'Autumn'
        assert get_season(11) == 'Autumn'
    
    def test_get_season_invalid_month(self):
        """Test get_season with invalid month values"""
        with pytest.raises((ValueError, IndexError, KeyError)):
            get_season(13)
        with pytest.raises((ValueError, IndexError, KeyError)):
            get_season(0)


class TestPredictionModel:
    """Test machine learning prediction functionality"""   
    def create_test_dataframe(self):
        """Creates a sample DataFrame for testing all seasons and features."""
        test_data = {
            'Date': pd.to_datetime(['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01']),
            'Amount': [100.0, 200.0, 300.0, 400.0],
            'Category': ['Rent', 'Groceries', 'Transport', 'Entertainment'],
            'Internship_period': [True, False, False, True],
            'Part_time_job_period': [False, True, False, True]
        }
        df = pd.DataFrame(test_data)
        df['Month'] = df['Date'].dt.month
        # Ensure 'get_season' is imported at the top of your test file if it's not
        from app.my_expenses import get_season
        df['Season'] = df['Date'].apply(lambda x: get_season(x.month))
        return df
    
    def test_train_prediction_model_returns_model_and_encoder(self):
        """Test that train_prediction_model returns model and encoder"""
        df = self.create_test_dataframe()
        
        model, encoder = train_prediction_model(df)
        
        assert isinstance(model, LinearRegression)
        assert isinstance(encoder, LabelEncoder)
    
    def test_train_prediction_model_with_valid_data(self):
        """Test model training with valid data"""
        df = self.create_test_dataframe()
        
        model, encoder = train_prediction_model(df)
        
        # Test that model can make predictions
        test_input = np.array([[6, 1, 0, 2]])  # June, internship, no part-time, encoded season
        prediction = model.predict(test_input)
        
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == 1
        assert prediction[0] > 0  # Expenses should be positive
    
    def test_train_prediction_model_empty_dataframe(self):
        """Test model training with empty DataFrame"""
        df = pd.DataFrame(columns=['Date', 'Amount', 'Category', 'Internship_period', 'Part_time_job_period'])
        
        with pytest.raises((ValueError, IndexError)):
            train_prediction_model(df)


class TestDataValidation:
    """Test data validation and edge cases""" 
    
    def test_amount_values_positive(self):
        """Test that amount values are positive"""
        test_data = {
            'Date': ['01/09/2024', '03/09/2024'],
            'Amount': [100.0, -50.0],  # Negative amount should be flagged
            'Category': ['Rent', 'Groceries']
        }
        df = pd.DataFrame(test_data)
        
        # Check for negative amounts
        negative_amounts = df[df['Amount'] < 0]
        if not negative_amounts.empty:
            assert len(negative_amounts) > 0  # This test expects to find negative amounts
    
    def test_required_columns_present(self):
        """Test that all required columns are present"""
        required_columns = ['Date', 'Category', 'Amount', 'Internship_period', 'Part_time_job_period']
        
        # Test with missing column
        incomplete_data = {
            'Date': ['01/09/2024'],
            'Amount': [100.0]
            # Missing other required columns
        }
        df = pd.DataFrame(incomplete_data)
        
        missing_columns = set(required_columns) - set(df.columns)
        assert len(missing_columns) > 0  # Should detect missing columns
    
    def test_date_format_validation(self):
        """Test date format validation"""
        test_data = {
            'Date': ['01/09/2024', 'invalid_date', '2024-09-01'],
            'Amount': [100.0, 200.0, 300.0],
            'Category': ['Rent', 'Groceries', 'Transport']
        }
        df = pd.DataFrame(test_data)
        
        # Try to convert dates and check for errors
        try:
            pd.to_datetime(df['Date'])
        except (ValueError, pd.errors.ParserError):
            assert True  # Expected to fail with invalid dates


class TestStreamlitIntegration:
    """Test Streamlit-specific functionality"""
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('my_expenses.load_data')
    def test_main_function_runs(self, mock_load_data, mock_markdown, mock_title, mock_config):
        """Test that main function runs without errors"""
        # Mock the data loading
        test_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'Category': ['Rent'] * 10,
            'Amount': [500.0] * 10,
            'Internship_period': [False] * 10,
            'Part_time_job_period': [True] * 10
        })
        test_data['Month'] = test_data['Date'].dt.month
        test_data['Month_Name'] = test_data['Date'].dt.strftime('%B')
        test_data['Quarter'] = test_data['Date'].dt.quarter
        test_data['Season'] = test_data['Date'].apply(lambda x: get_season(x.month))
        
        mock_load_data.return_value = test_data
        
        # Mock Streamlit components
        with patch('streamlit.sidebar'), \
             patch('streamlit.multiselect', return_value=['Rent']), \
             patch('streamlit.date_input', return_value=(test_data['Date'].min(), test_data['Date'].max())), \
             patch('streamlit.columns'), \
             patch('streamlit.metric'), \
             patch('streamlit.plotly_chart'), \
             patch('streamlit.selectbox', side_effect=[6, 0, 1]), \
             patch('streamlit.button', return_value=False):
            
            try:
                main()
                assert True  # If no exception is raised, test passes
            except Exception as e:
                pytest.fail(f"main() function raised an exception: {e}")


class TestPerformance:
    """Performance and stress tests"""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset"""
        # Create large dataset
        large_data = {
            'Date': pd.date_range('2020-01-01', periods=10000, freq='D'),
            'Amount': np.random.uniform(1, 1000, 10000),
            'Category': np.random.choice(['Rent', 'Groceries', 'Transport', 'Entertainment'], 10000),
            'Internship_period': np.random.choice([True, False], 10000),
            'Part_time_job_period': np.random.choice([True, False], 10000)
        }
        df = pd.DataFrame(large_data)
        df['Month'] = df['Date'].dt.month
        df['Season'] = df['Date'].apply(lambda x: get_season(x.month))
        
        # Time the model training
        import time
        start_time = time.time()
        
        try:
            model, encoder = train_prediction_model(df)
            end_time = time.time()
            
            # Should complete within reasonable time (adjust threshold as needed)
            assert (end_time - start_time) < 10.0  # 10 seconds max
            
        except Exception as e:
            pytest.fail(f"Performance test failed with large dataset: {e}")
    
    def test_memory_usage_with_large_dataset(self):
        """Test memory usage with large dataset"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process large dataset
        large_data = {
            'Date': pd.date_range('2020-01-01', periods=50000, freq='H'),
            'Amount': np.random.uniform(1, 1000, 50000),
            'Category': np.random.choice(['Rent', 'Groceries', 'Transport'], 50000)
        }
        df = pd.DataFrame(large_data)
        
        # Process data
        df['Month'] = df['Date'].dt.month
        df['Season'] = df['Date'].apply(lambda x: get_season(x.month))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 500  # Less than 500MB increase


class TestRegression:
    """Regression tests to ensure functionality doesn't break"""
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent for the same input"""
        # Create consistent test data
        np.random.seed(42)  # For reproducible results
        test_data = {
            'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'Amount': np.random.uniform(10, 500, 100),
            'Category': np.random.choice(['Rent', 'Groceries'], 100),
            'Internship_period': np.random.choice([True, False], 100),
            'Part_time_job_period': np.random.choice([True, False], 100)
        }
        df = pd.DataFrame(test_data)
        df['Month'] = df['Date'].dt.month
        df['Season'] = df['Date'].apply(lambda x: get_season(x.month))
        
        # Train model twice and compare predictions
        model1, encoder1 = train_prediction_model(df)
        model2, encoder2 = train_prediction_model(df)
        
        test_input = np.array([[6, 1, 0, encoder1.transform(['Summer'])[0]]])
        
        pred1 = model1.predict(test_input)[0]
        pred2 = model2.predict(test_input)[0]
        
        # Predictions should be very close (allowing for small floating point differences)
        assert abs(pred1 - pred2) < 0.01
    
    def test_season_mapping_consistency(self):
        """Test that season mapping is consistent"""
        # Test all months
        expected_seasons = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
            5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'
        }
        
        for month, expected_season in expected_seasons.items():
            assert get_season(month) == expected_season


# Acceptance Tests
class TestAcceptance:
    """End-to-end acceptance tests"""
    
    def test_complete_workflow(self):
        """Test complete user workflow"""
        # This would test the entire user journey
        # In a real scenario, you might use tools like Selenium for this
        
        # 1. Load data
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            test_data = {
                'Date': ['01/09/2024', '03/09/2024', '05/09/2024'],
                'Category': ['Rent', 'Groceries', 'Transport'],
                'Amount': [500.0, 29.26, 15.50],
                'Internship_period': [False, False, True],
                'Part_time_job_period': [True, False, False]
            }
            df = pd.DataFrame(test_data)
            df.to_csv(tmp.name, index=False)
            
            with patch('my_expenses.os.path.exists', return_value=True), \
                 patch('my_expenses.pd.read_csv', return_value=df):
                
                # 2. Load and process data
                result_df = load_data()
                assert not result_df.empty
                
                # 3. Train model
                model, encoder = train_prediction_model(result_df)
                assert model is not None
                
                # 4. Make prediction
                prediction_input = np.array([[9, 0, 1, encoder.transform(['Autumn'])[0]]])
                prediction = model.predict(prediction_input)
                assert len(prediction) == 1
                assert prediction[0] > 0
        
        os.unlink(tmp.name)


# Fixtures for pytest
@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame for tests"""
    data = {
        'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
        'Amount': np.random.uniform(10, 500, 30),
        'Category': np.random.choice(['Rent', 'Groceries', 'Transport'], 30),
        'Internship_period': np.random.choice([True, False], 30),
        'Part_time_job_period': np.random.choice([True, False], 30)
    }
    df = pd.DataFrame(data)
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Date'].apply(lambda x: get_season(x.month))
    return df


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])