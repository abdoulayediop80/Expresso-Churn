
# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

# Créez le répertoire models s'il n'existe pas
os.makedirs('models', exist_ok=True)

# Charger les données
data = pd.read_csv('C:\\Users\\Ali\\Documents\\Expresso Churn\\data\\Expresso_churn_dataset.csv')

# Convertir les colonnes catégorielles en valeurs numériques
data['REGION'] = LabelEncoder().fit_transform(data['REGION'].astype(str))
data['TENURE'] = LabelEncoder().fit_transform(data['TENURE'].astype(str))
data['ARPU_SEGMENT'] = LabelEncoder().fit_transform(data['ARPU_SEGMENT'].astype(str))

# Sélectionner les caractéristiques et la variable cible
X = data[['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'DATA_VOLUME']]
y = data['CHURN']

# Imputer les valeurs manquantes
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size= 0.2, random_state=42)

# Entraîner le modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Sauvegarder le modèle
model_path = 'models/logistic_model.pkl'
joblib.dump(model, model_path)

print(f"Modèle sauvegardé à {model_path}")