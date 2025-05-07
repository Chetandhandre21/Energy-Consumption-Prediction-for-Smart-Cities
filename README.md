# Energy-Consumption-Prediction-for-Smart-Cities

This project is a Streamlit-based Energy Consumption Prediction Dashboard that integrates OLAP operations and machine learning predictions using a Random Forest Regressor. It connects to a MySQL database for data storage and retrieval.

Features:
OLAP Operations:

Slice and dice energy data by Zone and Date Range (Month/Day).
Aggregate energy consumption, temperature, and humidity.
Energy Prediction:

Predict energy consumption based on inputs like temperature, humidity, square footage, occupancy, HVAC usage, etc.
Estimate energy costs.
Store predictions in the database.
Setup:
Install dependencies:
Set up MySQL database and update the connection string in the code.
Run the app:
Outputs:
OLAP Dashboard: Aggregated energy data.
Predictions: Energy consumption and estimated cost stored in the database.
