import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/rohanaryagondi/Downloads/attention_detection_dataset_v2.csv")  # Replace with file path if running elsewhere

X = df.drop(columns=["label"])
y = df["label"]

# Separate non-numeric and numeric columns
categorical = ["pose"]
numeric = [col for col in X.columns if col not in categorical]

# Data prep
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric), # Z score normalize numeric columns
        ("cat", OneHotEncoder(drop='first', sparse_output=False) # One-hot encode categorical columns
         , categorical)
    ]
)

X_processed = preprocessor.fit_transform(X)

encoded_pose_names = preprocessor.named_transformers_["cat"].get_feature_names_out(["pose"])
all_feature_names = numeric + list(encoded_pose_names)

# Apply SVD
svd = TruncatedSVD(n_components=2) # Reduce to 2 components
svd_components = svd.fit_transform(X_processed)
component_matrix = svd.components_

feature_contributions = np.abs(component_matrix[0]) + np.abs(component_matrix[1])  # PC1 and PC2
feature_importance = pd.DataFrame({
    "Feature": all_feature_names,
    "Contribution to PC1 PC2": feature_contributions
}).sort_values(by="Contribution to PC1 PC2", ascending=False)

print("Feature Importance:")
print(feature_importance.to_string(index=False))

# We want the correlation between features only
df_no_label = df.drop(columns=["label"])

# Correlation matrix
correlation_matrix = df_no_label.corr(numeric_only=True)

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

