## Housing Price Prediction using Regression

This project demonstrates the process of predicting housing prices using machine learning models. The goal is to accurately estimate house prices based on key features like area, number of bedrooms, and other property attributes. Through a combination of exploratory data analysis, feature engineering, and model evaluation, this project provides a comprehensive approach to solving a real-world regression problem.

### Project highlights
- **Objective**: Predict the housing prices based on various features such as area, number of bedrooms, amenities, etc.

- **Techniques**:
	-   Data exploration to understand the dataset and its structure.
	-	Handling outliers and missing values for robust data preparation.
	-	Feature engineering, including creating new features and transforming existing ones.
	-	Model implementation and comparison, including Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
	- Evaluation using metrics like Mean Absolute Error (MAE) and R² Score.

- **Tools & Libraries**:
	-	Pandas, NumPy, Matplotlib, Seaborn, scikit-learn


### Results

The models achieve the following results on the testing set:
- Linear Regression:
	- R² Score: ~0.65
	- MAE: ~0.011
- Decision Tree Regressor:
	- R² Score: ~0.53
	- MAE: ~0.011
- Random Forest Regressor:
	- R² Score: ~0.72
	- MAE: ~0.0085

The Random Forest Regressor emerged as the best-performing model, providing a balance between accuracy and generalization.

### Visualizations

- Feature importance rankings from the Random Forest model, highlighting the most impactful features in predicting housing prices.
- Training and testing performance metrics for all models.
- Correlation heatmap to visualize relationships between features.


[Main page](/)