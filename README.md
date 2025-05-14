Stress and Health Metrics Analysis
Overview
This project analyzes health and lifestyle data to predict stress levels and identify correlations between various health metrics and stress. It categorizes occupations into professional fields, examines stress patterns across these fields, and uses machine learning to predict stress levels based on key health factors. Additionally, it provides personalized recommendations to improve specific health metrics based on ideal ranges.
Project Structure

Data Source:
stress_detection_data.csv: Contains health and lifestyle data, including stress levels, occupation, and numerical metrics like cholesterol, blood pressure, and sleep quality.


Main Script: Stress Data.ipynb (Jupyter Notebook)
Preprocesses data, categorizes occupations, analyzes stress patterns, trains machine learning models, and generates recommendations.


Dependencies:
Python 3.x
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, networkx, scipy


Outputs:
Visualizations:
Field of work similarity network colored by dominant stress level.
Heatmap of field similarity based on stress patterns.
Correlation matrix heatmap for numerical health metrics.
Plots of model accuracy vs. test size, KNN neighbors, and Random Forest estimators.


Console outputs:
Top 5 health metrics correlated with stress.
Optimal test size and model parameters for KNN and Random Forest.
Predicted stress level for a sample patient.
Personalized recommendations for improving selected health metrics.





Key Features

Occupation Categorization:

Groups occupations into 12 professional fields (e.g., Technology/IT, Healthcare/Medical) using predefined lists.
Adds a Field_of_work column to the dataset for analysis.


Stress Patterns by Field:

Creates a sparse matrix of stress level distributions across fields.
Computes cosine similarity between fields to identify similar stress profiles.
Builds an undirected weighted graph, filtering edges to the top 75% by weight, and colors nodes by dominant stress level (Low: green, Medium: orange, High: red).
Visualizes field similarities with a heatmap.


Health Metrics Correlation:

Maps stress levels to numerical values (Low: 1, Medium: 2, High: 3).
Computes and visualizes a correlation matrix for numerical health metrics.
Identifies the top 5 metrics most correlated with stress.


Stress Prediction with Machine Learning:

Uses the top 5 correlated features to train K-Nearest Neighbors (KNN) and Random Forest models.
Optimizes test size and model parameters (KNN neighbors, Random Forest estimators) by evaluating accuracy across ranges.
Selects the best model (KNN or Random Forest) based on accuracy and predicts stress for a sample patient.


Personalized Health Recommendations:

Defines ideal ranges for key health metrics (e.g., cholesterol < 200 mg/dl, stress = Low).
Filters data for individuals with ideal metric values and calculates average values for correlated features.
Compares user-inputted metrics to ideal averages and provides recommendations (e.g., "Decrease screen time to 3 hours/day").



Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/stress-health-analysis.git
cd stress-health-analysis


Install Dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn networkx scipy


Prepare Data:

Place stress_detection_data.csv in the project directory.
Ensure the file includes columns like Occupation, Stress_Detection, and numerical health metrics (e.g., Cholesterol_Level, Blood_Pressure).


Run the Analysis:

Open Stress Data.ipynb in Jupyter Notebook or Google Colab.
Execute the cells sequentially to process the data and generate outputs.
Alternatively, convert the notebook to a Python script:jupyter nbconvert --to script "Stress Data.ipynb"
python "Stress Data.py"





Usage

Run the notebook to generate visualizations, model results, and recommendations.
Modify the occupation_categories dictionary to adjust job field groupings.
Adjust the edge weight threshold (75th percentile) or ideal health ranges to explore different scenarios.
Input custom health metric values when prompted to receive personalized recommendations.

Results

Stress by Field: Fields with similar stress profiles are visualized in a network, with node colors indicating dominant stress levels.
Correlations: The top 5 health metrics most correlated with stress (e.g., cholesterol, blood pressure) are identified.
Model Performance: The best model (KNN or Random Forest) achieves optimal accuracy (e.g., ~0.85) with tuned parameters (e.g., 5 neighbors or 50 estimators).
Sample Prediction: A sample patient’s stress level is predicted based on input metrics (e.g., cholesterol: 180 mg/dl, blood pressure: 180 mmHg).
Recommendations: Users receive tailored advice, such as reducing caffeine intake to match ideal averages for low stress.

Limitations

The dataset is limited to the provided CSV and may not generalize across populations.
Assumes consistent and accurate data for health metrics and stress levels.
Correlation-based feature selection may overlook complex interactions between variables.
Interactive user input for recommendations requires manual entry, which may be impractical for large-scale use.

Future Improvements

Incorporate additional datasets to enhance generalizability.
Use advanced feature selection methods (e.g., mutual information) to capture nonlinear relationships.
Automate user input collection via a web interface or API.
Add cross-validation to improve model robustness and prevent overfitting.
Explore deep learning models for more complex stress prediction tasks.

