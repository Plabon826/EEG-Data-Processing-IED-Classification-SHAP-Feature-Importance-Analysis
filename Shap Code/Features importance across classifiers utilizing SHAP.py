import pandas as pd
import matplotlib.pyplot as plt
import re  # import regex library

# Extract base feature names by splitting on numeric numbers and taking the part after the number
shap_importance_test['Base Feature'] = shap_importance_test['Feature'].apply(
    lambda x: re.split(r'\d+', x)[-1]
)

# Aggregate SHAP importance by summing over all leads for each base feature
agg_shap_importance = shap_importance_test.groupby('Base Feature')['SHAP Importance (%)'].sum().reset_index()

# Sort features by aggregated importance descending
agg_shap_importance = agg_shap_importance.sort_values(by='SHAP Importance (%)', ascending=False)

# Print aggregated SHAP importances
print("Aggregated SHAP Importance Across All Leads:")
print(agg_shap_importance)

# Plot aggregated importance
plt.figure(figsize=(10, 6))
agg_shap_importance.plot(kind='bar', x='Base Feature', y='SHAP Importance (%)', legend=False)
plt.title('Aggregated SHAP Feature Importance Across All Leads')
plt.xlabel('Feature (Aggregated over leads)')
plt.ylabel('Total SHAP Importance (%)')
plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()