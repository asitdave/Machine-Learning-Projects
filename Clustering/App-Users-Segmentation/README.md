## User Segmentation and App Uninstall Prediction

This project analyzes user behavior on a mobile application to segment users into different categories and predict whether they will uninstall the app. Using K-Means clustering, we group users based on behavioral patterns like screen time, spending, and engagement. We then apply Random Forest and XGBoost models to predict app uninstallation, providing insights into user retention strategies.


### Project highlights
- **Objective**:
	- Identify key factors influencing user retention and churn.
	- Segment users based on behavior using clustering techniques.
	- Train machine learning models to predict app uninstallation.
	- Implement an interactive system for real-time user predictions.

- **Techniques Used**:
	- User Segmentation: Clustering with K-Means to categorize user behaviors.
	- Churn Prediction: Classification using Random Forest and XGBoost.

- **Tools & Libraries**:
	- Machine Learning: K-Means, scikit-learn, XGBoost
	- Data Visualization: Plotly, Seaborn, Matplotlib


### Results
**User Segmentation**:
- Retained Users: Moderate to high engagement and spending.
- Churn Users: Low engagement and minimal spending.
- Needs Attention: Mixed behaviors, at risk of churn.

**Churn Prediction**:
- Random Forest Accuracy: 97.75%
- XGBoost Accuracy: 97.62%
- Last Visited Minutes is the strongest predictor of churn.
- XGBoost has better recall for detecting all uninstallations.
- Random Forest provides a more balanced prediction.


[Main page](/)