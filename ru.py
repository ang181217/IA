import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
import pandas as pd

data = pd.read_csv('ru.csv')

print(data.head())

X = data[['Overall scores', 'Research Quality Score', 'Industry Score', 'International Outlook', 'Research Environment Score', 'Teaching Score']]  
y = data['rank'] 

pipeline = make_pipeline(StandardScaler(), LogisticRegression())

param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Mejores hiperparámetros:", grid.best_params_)

joblib.dump(best_model, 'modelo_entrenado_regresion_logistica.pkl')

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nPrecisión del modelo en el conjunto de prueba:", accuracy)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
