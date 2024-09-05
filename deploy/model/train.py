import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_excel(r'C:\Users\Admin\Downloads\flyPrediction\deploy\model\Trained-group.xlsx')

# Drop unnecessary columns
df = df.drop(columns=['ID', 'Genus', 'No', 'Sex'])

# Feature columns
shape_columns = ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)', 'e(6-7)', 'f(7-10)', 
                 'g(9-10)', 'h(9-15)', 'i(15-16)', 'j(14-15)', 'k(13-14)', 
                 'l(13-17)', 'm(17-18)', 'n(1-18)', 'o(2-13)', 'p(3-12)', 
                 'q(12-13)', 'r(5-12)', 's(11-14)', 't(8-11)', 'u(7-8)', 
                 'v(8-9)', 'w(11-12)']

categorical_columns = ['Gena color', 'Body color']

# Rescale function
def rescale_abcd(row):
    cols = ['a(1-2)', 'b(2-3)', 'e(6-7)', 'g(9-10)']
    total = sum(row[cols])
    if total != 0:
        factor = 5000 / total
        for col in cols:
            row[col] *= factor
    return row

# Apply rescaling to specific columns
df = df.apply(rescale_abcd, axis=1)

# One-hot encoding for categorical columns
df = pd.get_dummies(df, columns=categorical_columns)

# Train Family Model (uses both shape and categorical features)
X_family = df[shape_columns + list(df.filter(regex='Gena color|Body color'))]
y_family = df['Family']

X_train_family, X_test_family, y_train_family, y_test_family = train_test_split(X_family, y_family, test_size=0.2, random_state=42)

family_model = DecisionTreeClassifier(random_state=42)
family_model.fit(X_train_family, y_train_family)

y_pred_family = family_model.predict(X_test_family)
print(f"Family Model Accuracy: {accuracy_score(y_test_family, y_pred_family):.4f}")
print(f"Family Model Report:\n{classification_report(y_test_family, y_pred_family)}")

# Train Species Model (uses only shape features)
X_species = df[shape_columns]
y_species = df['Species']

X_train_species, X_test_species, y_train_species, y_test_species = train_test_split(X_species, y_species, test_size=0.2, random_state=42)

species_model = DecisionTreeClassifier(random_state=42)
species_model.fit(X_train_species, y_train_species)

y_pred_species = species_model.predict(X_test_species)
print(f"Species Model Accuracy: {accuracy_score(y_test_species, y_pred_species):.4f}")
print(f"Species Model Report:\n{classification_report(y_test_species, y_pred_species)}")

# Save models
joblib.dump(family_model, 'family_model.pkl')
joblib.dump(species_model, 'species_model.pkl')
