import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import pandas.errors
import requests
import json


# Load trained model
model = joblib.load('student_model.pkl')

# Google API Configuration
GOOGLE_API_KEY = "AIzaSyB1ImlCwONtmPztkQ4IVOklr5cANt7cbZk"

# Define input feature names (must match training)
features = [
    'hint_count', 'bottom_hint', 'attempt_count', 'ms_first_response', 'duration',
    'Average_confidence(FRUSTRATED)', 'Average_confidence(CONFUSED)',
    'Average_confidence(CONCENTRATING)', 'Average_confidence(BORED)',
    'action_count', 'hint_dependency', 'response_speed', 'confidence_balance',
    'engagement_ratio', 'efficiency_indicator'
]

# 🧠 Import helper functions
from helper_functions import (
    categorize_student_performance,
    recommend_learning_material,
    generate_feedback_message,
    generate_learner_profile,
    generate_combined_recommendation,
    get_google_learning_resources,
    generate_ai_enhanced_feedback
)


# 🌐 Streamlit App Interface
st.title("🎓 AI Tutor: Student Performance Predictor")
st.markdown("**Powered by AI and Google API Integration**")

# Sidebar for additional features
st.sidebar.title("🔧 Settings & Tools")
st.sidebar.markdown("---")

# API Status
st.sidebar.subheader("🔑 API Status")
st.sidebar.success("✅ Google API Key: Active")
st.sidebar.info(f"🔗 API Key: {GOOGLE_API_KEY[:20]}...")

# Quick Stats
st.sidebar.subheader("📊 Quick Stats")
if os.path.exists("prediction_log.csv"):
    try:
        log_data = pd.read_csv("prediction_log.csv")
        total_predictions = len(log_data)
        unique_students = log_data['student_id'].nunique()
        st.sidebar.metric("Total Predictions", total_predictions)
        st.sidebar.metric("Unique Students", unique_students)
    except:
        st.sidebar.metric("Total Predictions", 0)
        st.sidebar.metric("Unique Students", 0)
else:
    st.sidebar.metric("Total Predictions", 0)
    st.sidebar.metric("Unique Students", 0)

st.sidebar.markdown("---")

# Main content
st.subheader("👤 Student Information")
student_id = st.text_input("Enter Student ID", "", help="Enter a unique identifier for the student")

st.markdown("### 📊 Enter Student Behavior Data")

# Organize inputs into tabs for better UX
tab1, tab2, tab3 = st.tabs(["🎯 Basic Metrics", "😊 Emotional States", "🧠 Advanced Features"])

user_input = {}

with tab1:
    st.markdown("**Basic Learning Metrics**")
    col1, col2 = st.columns(2)
    
    with col1:
        user_input['hint_count'] = st.slider("Hint Count", 0, 20, 5, help="Number of hints used")
        user_input['bottom_hint'] = st.slider("Bottom Hints", 0, 20, 5, help="Hints from bottom of screen")
        user_input['attempt_count'] = st.slider("Attempt Count", 0, 15, 3, help="Number of attempts made")
    
    with col2:
        user_input['ms_first_response'] = st.slider("First Response Time (ms)", 100, 3000, 800, help="Time to first response")
        user_input['duration'] = st.slider("Session Duration (ms)", 100, 3000, 1000, help="Total session time")
        user_input['action_count'] = st.slider("Action Count", 0.0, 1.0, 0.5, help="Number of actions taken")

with tab2:
    st.markdown("**Emotional State Confidence Levels**")
    col1, col2 = st.columns(2)
    
    with col1:
        user_input['Average_confidence(FRUSTRATED)'] = st.slider("😤 Frustrated", 0.0, 1.0, 0.2, help="Confidence level for frustration")
        user_input['Average_confidence(CONFUSED)'] = st.slider("😕 Confused", 0.0, 1.0, 0.2, help="Confidence level for confusion")
    
    with col2:
        user_input['Average_confidence(CONCENTRATING)'] = st.slider("🎯 Concentrating", 0.0, 1.0, 0.6, help="Confidence level for concentration")
        user_input['Average_confidence(BORED)'] = st.slider("😴 Bored", 0.0, 1.0, 0.1, help="Confidence level for boredom")

with tab3:
    st.markdown("**Advanced Behavioral Features**")
    col1, col2 = st.columns(2)
    
    with col1:
        user_input['hint_dependency'] = st.slider("Hint Dependency", 0.0, 1.0, 0.2, help="Reliance on hints")
        user_input['response_speed'] = st.slider("Response Speed", 100, 3000, 900, help="Speed of responses")
    
    with col2:
        user_input['confidence_balance'] = st.slider("Confidence Balance", 0.0, 1.0, 0.5, help="Balance of confidence levels")
        user_input['engagement_ratio'] = st.slider("Engagement Ratio", 0.0, 1.0, 0.5, help="Level of engagement")
        user_input['efficiency_indicator'] = st.slider("Efficiency Indicator", 0.0, 1.0, 0.5, help="Learning efficiency")


# 🧠 Predict button logic
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🚀 Predict Performance", type="primary", use_container_width=True)

if predict_button:
    # Add progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Analyzing student data...")
    progress_bar.progress(0.25)
    
    # Ensure features are in the correct order for the model
    input_df = pd.DataFrame([user_input])[features]
    predicted_score = model.predict(input_df)[0]
    
    status_text.text("🧠 Generating learner profile...")
    progress_bar.progress(0.5)
    
    learner_profile = generate_learner_profile(user_input)
    
    status_text.text("🤖 AI-enhanced analysis in progress...")
    progress_bar.progress(0.75)


    # Categorize student
    cat_num, cat_name, desc, emoji = categorize_student_performance(predicted_score)
    
    combined_recommendation = generate_combined_recommendation(cat_name, learner_profile)
    rec = recommend_learning_material(cat_num)
    fb = generate_ai_enhanced_feedback(cat_name, learner_profile, predicted_score)
    google_resources = get_google_learning_resources("student performance", cat_name)
    
    status_text.text("✅ Analysis complete!")
    progress_bar.progress(1.0)
    status_text.empty()

    # 📊 Display results with enhanced formatting
    st.markdown("---")
    st.subheader("📋 AI-Powered Prediction Results")
    
    # Main metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Score", f"{predicted_score:.3f}", help="Predicted correctness score (0-1)")
    
    with col2:
        confidence_level = "High" if abs(predicted_score - 0.5) > 0.2 else "Medium"
        st.metric("Confidence", confidence_level, help="Prediction confidence level")
    
    with col3:
        st.metric("Category", f"{emoji} {cat_name}", help="Performance category")
    
    # Detailed results in expandable sections
    with st.expander("🎯 Performance Analysis", expanded=True):
        st.write(f"**Learning Profile:** {learner_profile}")
        st.write(f"**Performance Level:** {desc}")
        st.write(f"**Category:** {emoji} {cat_name}")
    
    with st.expander("💬 AI-Enhanced Feedback"):
        st.write(fb)
    
    with st.expander("📚 Learning Recommendations"):
        st.write(f"**Combined Recommendation:** {combined_recommendation}")
        st.write(f"**Google-Enhanced Resources:** {google_resources}")
    
    # Visual indicators
    st.markdown("### 📊 Performance Indicators")
    
    # Progress bar for predicted score
    st.markdown("**Predicted Correctness Score:**")
    st.progress(float(predicted_score))
    
    # Color-coded performance level
    if predicted_score < 0.3:
        st.error(f"🆘 Needs Immediate Support - Score: {predicted_score:.3f}")
    elif predicted_score < 0.6:
        st.warning(f"⚠️ Room for Improvement - Score: {predicted_score:.3f}")
    else:
        st.success(f"✅ Good Performance - Score: {predicted_score:.3f}")



    # Add timestamp + user data + result to a row
    log_row = {
        "student_id": student_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **user_input,
        "predicted_score": predicted_score,
        "category": cat_name,
        "learner_profile": learner_profile
    }
    
    # Create CSV with headers if it doesn't exist or is empty
    log_path = "prediction_log.csv"
    required_columns = ["student_id", "timestamp", *features, "predicted_score", "category", "learner_profile"]

    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        pd.DataFrame(columns=required_columns).to_csv(log_path, index=False)


    # Append to CSV file
    log_df = pd.DataFrame([log_row])
    log_df.to_csv("prediction_log.csv", mode='a', header=not os.path.exists("prediction_log.csv"), index=False)
    
    
    st.subheader("📊 Student Progress Over Time")
    # Check if file exists
    if os.path.exists("prediction_log.csv") and student_id:
        try:
            log_data = pd.read_csv("prediction_log.csv")
            log_data['timestamp'] = pd.to_datetime(log_data['timestamp']) # Convert timestamp to datetime


            # student history filter
            student_history = log_data[log_data['student_id'] == student_id]
            
            if not student_history.empty:
                st.subheader(f"📈 Progress for Student: {student_id}")
                st.line_chart(student_history.set_index("timestamp")["predicted_score"])
            else:
                st.subheader(f"📈 First-time Progress for Student: {student_id}")
                # Create a single-point chart to start the graph
                new_entry = pd.DataFrame({
                    "timestamp": [pd.Timestamp.now()],
                    "predicted_score": [predicted_score]
                }).set_index("timestamp")

                st.line_chart(new_entry["predicted_score"])
        except pandas.errors.EmptyDataError:
            st.warning("📁 prediction_log.csv exists but is empty. A new log entry will be added after this prediction.")
            
    else:
        st.info("No prediction history found yet.")
        
    # 🔍 Show full prediction history for this student
    if os.path.exists("prediction_log.csv") and student_id:
        log_data = pd.read_csv("prediction_log.csv")
        log_data['timestamp'] = pd.to_datetime(log_data['timestamp'])

        student_history = log_data[log_data['student_id'] == student_id]

        if not student_history.empty:
            st.subheader("📋 Prediction History Table")

            # Select only needed columns for display
            display_cols = ["timestamp", "predicted_score", "category", "learner_profile"]
            display_data = student_history[display_cols].sort_values(by="timestamp", ascending=False).reset_index(drop=True)

            st.dataframe(display_data, use_container_width=True)

    
    
st.markdown("---")
st.markdown("---")
st.subheader("🧑‍🏫 Teacher Dashboard: Class Overview")

# Load log file
if os.path.exists("prediction_log.csv"):
    log_data = pd.read_csv("prediction_log.csv")
    log_data['timestamp'] = pd.to_datetime(log_data['timestamp'])

    # Keep only the latest record per student
    latest_by_student = log_data.sort_values("timestamp", ascending=False).drop_duplicates("student_id", keep='first')

    # Optional filters
    selected_category = st.selectbox("Filter by Category", ["All"] + sorted(latest_by_student["category"].unique()))
    selected_profile = st.selectbox("Filter by Profile", ["All"] + sorted(latest_by_student["learner_profile"].unique()))

    filtered_data = latest_by_student.copy()
    if selected_category != "All":
        filtered_data = filtered_data[filtered_data["category"] == selected_category]
    if selected_profile != "All":
        filtered_data = filtered_data[filtered_data["learner_profile"] == selected_profile]

    # Display columns
    display_cols = ["student_id", "timestamp", "predicted_score", "category", "learner_profile"]
    st.dataframe(filtered_data[display_cols].sort_values("timestamp", ascending=False), use_container_width=True)

else:
    st.info("No prediction history available yet.")





# # Check if file exists
# if os.path.exists("prediction_log.csv") and student_id:
#     try:
#         log_data = pd.read_csv("prediction_log.csv")
#         log_data['timestamp'] = pd.to_datetime(log_data['timestamp']) # Convert timestamp to datetime


#         # student history filter
#         student_history = log_data[log_data['student_id'] == student_id]
        
#         if not student_history.empty:
#             st.subheader(f"📈 Progress for Student: {student_id}")
#             st.line_chart(student_history.set_index("timestamp")["predicted_score"])
#         else:
#             st.subheader(f"📈 First-time Progress for Student: {student_id}")
#             # Create a single-point chart to start the graph
#             new_entry = pd.DataFrame({
#                 "timestamp": [pd.Timestamp.now()],
#                 "predicted_score": [predicted_score]
#             }).set_index("timestamp")

#             st.line_chart(new_entry["predicted_score"])
#     except pandas.errors.EmptyDataError:
#         st.warning("📁 prediction_log.csv exists but is empty. A new log entry will be added after this prediction.")
        
# else:
#     st.info("No prediction history found yet.")

