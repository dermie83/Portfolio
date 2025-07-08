import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


try:
    df = pd.read_csv('Shipwreck_Classification_Analysis\Wrecks20240620_table_geocoded_processed.csv')
except FileNotFoundError:
    print("Error: 'Shipwreck_Classification_Analysis\Wrecks20240620_table_geocoded_processed.csv' not found.")
    exit()

print("Original DataFrame head:")
print(df.head())
print("\nOriginal DataFrame info:")
df.info()

df = df.drop(['location'], axis=1)


# Convert categorical features into numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Classification'],dtype=int, drop_first=True) # drop_first avoids multicollinearity
#  Calculate the correlation matrix
correlation_matrix = df.corr()
print("Correlation Matrix:\n", correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Features and Target Variable')
plt.show()

# Define X (features) and y (target)
X = df.drop(columns='casualties')
y = df['casualties']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# Scale numerical features (important for many models, including logistic regression when regularization is used)
scaler = StandardScaler()
numerical_cols = ['Date_of_Loss_Year_Only'] # Specify numerical columns to scale
X_resampled[numerical_cols] = scaler.fit_transform(X_resampled[numerical_cols])

print("\nFeatures (X) after scaling (head):")
print(X_resampled.head())
# 
# 3. Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# 4. Build and Train the 3 Models (Logistic Regression, Random Forest Classifier, Support Vector Machine (SVC))
model_logistic = LogisticRegression(solver='liblinear', random_state=42) # 'liblinear' is good for small datasets
model_logistic.fit(X_train, y_train)
print("\nLogistic Regression model trained.")

model_RF = RandomForestClassifier(n_estimators=100, random_state=42)
model_RF.fit(X_train, y_train)
print("\nRandom Forest Classifier model trained.")

model_SVC = SVC(random_state=42)
model_SVC.fit(X_train, y_train)
print("\nSVC model trained.")

models = {
    "Logistic Regression": model_logistic,
    "Random Forest Classifier": model_RF,
    "Support Vector Machine (SVC)": model_SVC
}

for model_name, model_pipeline in models.items():
    print(f"\n--- {model_name} Performance ---")
    y_pred = model_pipeline.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Casualties (0)', 'Casualties (1)']))

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)


    # Visualize the Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"\n--- {model_name} Confussion Matrix Performance ---")
    plt.show()

    if model_pipeline == model_logistic:
        print("\nCoefficients (for Logistic Regression):")
        # For binary classification, coef_ is typically (1, n_features)
        coef_df = pd.DataFrame(model_pipeline.coef_[0].T, index=X_resampled.columns, columns=['Coefficient'])
        print(coef_df.sort_values(by='Coefficient', ascending=False))
        print(f"Intercept: {model_pipeline.intercept_[0]:.4f}")
        # Save coefficients to CSV
        coef_filename = f"{model_name.lower().replace(' ', '_')}_coefficients.csv"
        coef_df.to_csv(coef_filename)
        print(f"Coefficients saved to {coef_filename}")

    elif model_pipeline == model_RF:
        print("\nFeature Importances (for Random Forest):")
        importance_df = pd.DataFrame(model_pipeline.feature_importances_, index=X_resampled.columns, columns=['Importance'])
        print(importance_df.sort_values(by='Importance', ascending=False))
        # Save feature importances to CSV
        importance_filename = f"{model_name.lower().replace(' ', '_')}_feature_importances.csv"
        importance_df.to_csv(importance_filename)
        print(f"Feature importances saved to {importance_filename}")
    else:
        print("\nThis model does not have 'coef_' or 'feature_importances_' attributes for direct interpretation.")
        print("Consider using model-agnostic methods like Permutation Importance or SHAP for feature insights.")


print("\nModel comparison complete. Analyze the reports above to compare performance.")



# # 5. Make Predictions
# y_pred = model_logistic.predict(X_test)
# y_pred_proba = model_logistic.predict_proba(X_test)[:, 1] # Probabilities of survival

# # 6. Evaluate the Model
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

# print(f"\nModel Accuracy: {accuracy:.4f}")
# print("\nConfusion Matrix:")
# print(conf_matrix)
# print("\nClassification Report:")
# print(class_report)

# # Visualize the Confusion Matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=['Predicted No', 'Predicted Yes'],
#             yticklabels=['Actual No', 'Actual Yes'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix for Ship Wrecks Casualties Prediction')
# plt.show()

# Optional: Display model coefficients
# print("\nModel Coefficients:")
# coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model_logistic.coef_[0]})
# coefficients.to_csv('coefficients.csv')
# print(coefficients.sort_values(by='Coefficient', ascending=False))
# print(f"Intercept: {model_logistic.intercept_[0]:.4f}")

# Interpretation of Coefficients:
# print("\n--- Interpretation of Coefficients ---")
# print("A positive coefficient indicates that as the feature value increases, the log-odds of casualties increase.")
# print("A negative coefficient indicates that as the feature value increases, the log-odds of casualties decrease.")