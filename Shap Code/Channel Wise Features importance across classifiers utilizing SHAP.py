import shap
from catboost import CatBoostClassifier
# Initialize SHAP TreeExplainer for the trained model
explainer = shap.TreeExplainer(cat)

# Compute SHAP values for the test data
shap_values_test = explainer.shap_values(X_test)

# Compute SHAP importance (mean absolute SHAP values per feature)
# For multi-class, shap_values_test is a list of 2D arrays (samples x features)
# We need to average across samples and then across classes
shap_importance_test_array = np.mean(np.abs(shap_values_test), axis=(0, -1))  # Average across samples and classes

# Normalize SHAP importance to sum to 100%
shap_importance_test_array = (shap_importance_test_array / shap_importance_test_array.sum()) * 100

# Validate that SHAP importance sums to 100%
total_importance_test = shap_importance_test_array.sum()
print(f"Total SHAP Importance (Testing set): {total_importance_test:.2f}%")  # Should print 100.00%

# Create a DataFrame for SHAP importance
shap_importance_test = pd.DataFrame({
    'Feature': X_test.columns,
    'SHAP Importance (%)': shap_importance_test_array  # Use the computed SHAP values
}).sort_values(by='SHAP Importance (%)', ascending=False)

# Get the top 10 features
top_10_features_test = shap_importance_test.head(10)

# Print the top 10 features
print("Top 10 Features by SHAP Importance (Testing Set):")
print(top_10_features_test)

# Plot the top 10 features
plt.figure(figsize=(10, 6))
top_10_features_test.plot(kind="bar", x='Feature', y='SHAP Importance (%)', legend=False)
plt.title("Top 10 SHAP Feature Importance (Testing Set)")
plt.xlabel("Feature")
plt.ylabel("Importance (%)")
plt.tight_layout()
plt.show()