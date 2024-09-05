import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import joblib
import os
import matplotlib.pyplot as plt

dataset_path = 'UNSW_NB15_testing-set.csv'
df = pd.read_csv(dataset_path)


features_columns_algorithm = ['id', 'proto', 'state', 'dur', 'sbytes', 'dbytes']
target_column_algorithm = 'proto'

X_algorithm = df[features_columns_algorithm]
y_algorithm = df[target_column_algorithm]


X_train_algorithm, X_test_algorithm, y_train_algorithm, y_test_algorithm = train_test_split(
    X_algorithm, y_algorithm, test_size=0.2, random_state=42
)


numerical_features_algorithm = ['id', 'dur', 'sbytes', 'dbytes']
categorical_features_algorithm = ['proto', 'state']

numeric_transformer_algorithm = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer_algorithm = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor_algorithm = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_algorithm, numerical_features_algorithm),
        ('cat', categorical_transformer_algorithm, categorical_features_algorithm)
    ])

model_algorithm = Pipeline(steps=[
    ('preprocessor', preprocessor_algorithm),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
model_algorithm.fit(X_train_algorithm, y_train_algorithm)


y_pred_algorithm = model_algorithm.predict(X_test_algorithm)
print("Algorithm Identification Model Evaluation:")
print(classification_report(y_test_algorithm, y_pred_algorithm))

df['predicted_proto'] = model_algorithm.predict(X_algorithm)


features_columns_security = ['predicted_proto', 'state']
target_column_security = 'strategy_label'


security_strategies_data = {
    'predicted_proto': ['tcp', 'udp', 'icmp', 'arp'],
    'state': ['FIN', 'INT', 'CON', 'REQ'],
    'strategy_label': ['Firewall Rules', 'Rate Limiting', 'Intrusion Detection', 'Access Control']
}
security_df = pd.DataFrame(security_strategies_data)

X_security = security_df[features_columns_security]
y_security = security_df[target_column_security]


X_train_security, X_test_security, y_train_security, y_test_security = train_test_split(
    X_security, y_security, test_size=0.2, random_state=42
)


categorical_features_security = ['predicted_proto', 'state']

categorical_transformer_security = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor_security = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer_security, categorical_features_security)
    ])


model_security = Pipeline(steps=[
    ('preprocessor', preprocessor_security),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
model_security.fit(X_train_security, y_train_security)


y_pred_security = model_security.predict(X_test_security)
print("\nSecurity Strategies Model Evaluation:")
print(classification_report(y_test_security, y_pred_security))

plt.figure(figsize=(10, 6))
sns.countplot(x=y_pred_security)
plt.title('Distribution of Predicted Security Strategies')
plt.xlabel('Security Strategy')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


if hasattr(model_security.named_steps['classifier'], 'feature_importances_'):
    importances = model_security.named_steps['classifier'].feature_importances_
    feature_names = model_security.named_steps['preprocessor'].get_feature_names_out()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title('Feature Importance for Security Strategy Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
plt.show()
