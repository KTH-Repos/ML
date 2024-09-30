# Define the test example
X_test = 12
Y_actual = 17

# Model 1
Y_pred_model1 = 2.14 + 1.22 * X_test
squared_error_model1 = (Y_pred_model1 - Y_actual) ** 2

# Model 2
Y_pred_model2 = (96.7499 - 41.4850 * X_test + 
                 6.6739 * X_test**2 - 
                 0.42864 * X_test**3 + 
                 0.0096354 * X_test**4)
squared_error_model2 = (Y_pred_model2 - Y_actual) ** 2

# Print results
print(f"Model 1 - Predicted Y: {Y_pred_model1:.2f}, Squared Error: {squared_error_model1:.2f}")
print(f"Model 2 - Predicted Y: {Y_pred_model2:.2f}, Squared Error: {squared_error_model2:.2f}")
