import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

# Read the files
df1 = pd.read_csv('England CSV.csv')
df2 = pd.read_csv('England 2 CSV.csv')

df = pd.concat([df1, df2], ignore_index=True)


# print(df.shape)
# print(df.head())


# pd.set_option('display.max_columns', None) <<<Show all columns>>>
# print(df1.head())
# print(df2.head())

def get_match_result(row):
    if row['FTH Goals'] > row['FTA Goals']:
        return 'H'  # Home Win
    elif row['FTH Goals'] < row['FTA Goals']:
        return 'A'  # Away Win
    else:
        return 'D'  # Draw


df['Match_Result'] = df.apply(get_match_result, axis=1)

columns_to_keep = [
    'HomeTeam', 'AwayTeam',
    'H Shots', 'A Shots',
    'H SOT', 'A SOT',
    'H Corners', 'A Corners',
    'H Fouls', 'A Fouls',
    'H Yellow', 'A Yellow',
    'H Red', 'A Red',
    'League',
    'Match_Result'
]

df = df[columns_to_keep]

# Initialize encoders
le_home = LabelEncoder()
le_away = LabelEncoder()

# Fit and transform HomeTeam and AwayTeam columns
df['HomeTeam'] = le_home.fit_transform(df['HomeTeam'])
df['AwayTeam'] = le_away.fit_transform(df['AwayTeam'])

le_league = LabelEncoder()
df['League'] = le_league.fit_transform(df['League'])

# le_result = LabelEncoder()  Otomatik atama yerine map kullanarak spesifik atama yapmayı tercih ettim
# df['Match_Result'] = le_result.fit_transform(df['Match_Result'])

df['Match_Result'] = df['Match_Result'].map({'H': 1, 'D': 0, 'A': 2})

df.to_csv('combined_matches.csv', index=False)

df.dropna(axis=0, how='any', inplace=True)
# print("Yeni shape:", df.shape)
#
# print(df.isnull().sum())


# Train & Test
x = df.drop('Match_Result', axis=1)
y = df['Match_Result']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ---------------------------------------------
# Logistic Regression
# ---------------------------------------------

model = LogisticRegression(max_iter=1000, class_weight='balanced')  # Modeli oluştur
model.fit(x_train, y_train)  # Eğit

y_pred_lr = model.predict(x_test)  # Tahmin yap

# Başarıyı ölç
print("\nLR Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nLR Classification Report:\n", classification_report(y_test, y_pred_lr))
print("\nLR Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# print("Toplam eksik değer sayısı:", df.isnull().sum().sum())
# print(df.isnull().sum())

print(y.value_counts())

# ---------------------------------------------
# Random Forest Classifier
# ---------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)

print("\nRF Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRF Classification Report:\n", classification_report(y_test, y_pred_rf))
print("\nRF Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# ---------------------------------------------
# XGBoost
# ---------------------------------------------

xgb_model = XGBClassifier(learning_rate=0.05,
                          n_estimators=500,
                          max_depth=6,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          random_state=42,
                          n_jobs=-1,
                          eval_metric='mlogloss')
xgb_model.fit(x_train, y_train)

# Tahmin
y_pred_xgb = xgb_model.predict(x_test)

# Performans
print("\nXGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("\nXGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Modelin temel hali
xgb_base = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=42
)

# Test edilecek parametre kombinasyonları
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Grid Search başlat
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    cv=3,
    verbose=1
)

# Ağırlık hesapla
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weight_dict = dict(zip(np.unique(y_train), class_weights))
sample_weights = y_train.map(weight_dict)

# GridSearchCV’e sample_weight geçmek için parametre hazırlığı
fit_params = {"sample_weight": sample_weights}

grid_search.fit(x_train, y_train, **fit_params)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi doğruluk (train):", grid_search.best_score_)

# En iyi modelle test verisi üzerinde değerlendirme
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(x_test)

print("XGBoost (GridSearch sonrası) Test Doğruluğu:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
