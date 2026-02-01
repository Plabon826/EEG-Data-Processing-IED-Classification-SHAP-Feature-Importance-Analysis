import pandas as pd
import re

# Example: shap_importance_test DataFrame with columns 'Feature' and 'SHAP Importance (%)'
# shap_importance_test = pd.DataFrame(...)

# Function to extract lead number from feature name
def extract_lead(feature_name):
    match = re.match(r"lead(\d+)_", feature_name)
    if match:
        return int(match.group(1))
    else:
        return None

# Apply the function to create a 'Lead' column
shap_importance_test['Lead'] = shap_importance_test['Feature'].apply(extract_lead)

# Group by 'Lead' and sum SHAP importance, ignoring features without lead numbers
lead_shap_importance = shap_importance_test.dropna(subset=['Lead']).groupby('Lead')['SHAP Importance (%)'].sum().reset_index()

# Sort descending by importance
lead_shap_importance = lead_shap_importance.sort_values(by='SHAP Importance (%)', ascending=False)

# Print the lead-wise SHAP importance
print(lead_shap_importance)

# Optional: plot the lead-wise importance
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar(lead_shap_importance['Lead'].astype(str), lead_shap_importance['SHAP Importance (%)'])
plt.xlabel('Lead')
plt.ylabel('Total SHAP Importance (%)')
plt.title('Lead-wise SHAP Feature Importance')
plt.tight_layout()
plt.show()