# AI Tutor: Student Performance Predictor

An AI-powered web application that predicts student performance based on behavioral and emotional data using machine learning.

## Features

- 🎯 **Student Performance Prediction**: Uses XGBoost model to predict correctness scores
- 🧠 **Learner Profile Classification**: Identifies learning styles (Fast but Careless, Slow and Careful, Confused Learner, Focused Performer)
- 📊 **Real-time Analytics**: Live progress tracking and historical data visualization
- 🤖 **AI-Enhanced Feedback**: Personalized recommendations and insights
- 🔗 **Google API Integration**: Enhanced learning resource recommendations
- 👨‍🏫 **Teacher Dashboard**: Class overview and student filtering

## Requirements

- Python 3.8+
- Streamlit
- XGBoost
- scikit-learn
- pandas
- joblib
- requests

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter a Student ID
2. Adjust the behavioral and emotional parameters using the sliders
3. Click "Predict Performance" to get AI-powered insights
4. View detailed analysis, recommendations, and progress tracking

## Model Information

- **Algorithm**: XGBoost Regressor
- **Features**: 15 behavioral and emotional indicators
- **Output**: Correctness score (0.0-1.0) with performance categorization
