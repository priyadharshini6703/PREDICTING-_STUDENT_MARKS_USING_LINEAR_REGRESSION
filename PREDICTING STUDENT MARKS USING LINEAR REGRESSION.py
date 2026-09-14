# STEP 1: IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


#STEP 2: LOAD DATASET
data = pd.read_csv("C:\\Users\\Priyadharshini R\\Desktop\Student_Marks.csv")
print(" Dataset Loaded!\n")
print(data.head())


# STEP 4: DEFINE X and y
X = data[['number_courses', 'time_study']]
y = data['Marks']


# STEP 5: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n Data Split Completed!")


# STEP 6: TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)
print(" Model Training Completed!")


# STEP 7: PREDICTION
y_pred = model.predict(X_test)

print("\n Sample Predictions:")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]:.2f}  |  Predicted: {y_pred[i]:.2f}")



# STEP 8: EVALUATION
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Performance:")
print("MAE:", mae)
print("R2 Score:", r2)


# STEP 9: PLOTTING
# Take limited data for better visualization
sample_data = data.sample(n=50, random_state=42)

# 1. Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred,color="green")
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.show()


# STEP 10: CUSTOM PREDICTION
# Example: 5 courses, 6 hours study
new_data = [[5, 6]]

predicted_marks = model.predict(new_data)

print("\n Predicted Marks:", predicted_marks[0])