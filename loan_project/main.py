import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Завантаження даних
data = pd.read_csv('loan_data.csv')

# Перевірка на відсутні значення
print(data.isnull().sum())

# Підготовка даних
X = data.drop(columns=['Loan_Status', 'Loan_ID', 'Gender', 'Married', 'Education'])   # Видалення стовпців
y = data['Loan_Status']

# Розділення на тренувальний та тестовий набір
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обробка числових та категоріальних даних
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Створення pipeline для числових та категоріальних даних
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Заповнення пропусків середнім
    ('scaler', StandardScaler())  # Масштабування числових даних
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Заповнення пропусків найбільш частим значенням
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Перетворення категоріальних даних в one-hot encoding
])

# Обробка всіх стовпців
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Створення моделі
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Налаштування параметрів для GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Виконання пошуку найкращих параметрів
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Виведення найкращих параметрів і точності
print(f'Найкращі параметри: {grid_search.best_params_}')
print(f'Найкращий результат перехресної валідації: {grid_search.best_score_}')

# Оцінка на тестових даних
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f'Точність на тестових даних після GridSearchCV: {test_accuracy}')

# Перехресна валідація для моделі (на основі навчальних даних)
cross_val_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

# Виведення середньої точності за результатами перехресної валідації
print(f'Середня точність за результатами перехресної валідації: {cross_val_scores.mean()}')

# Збереження моделі
# with open('loan_project/loan_app/utils/best_model.pkl', 'wb') as file:
#     pickle.dump(best_model, file)
