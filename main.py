import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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

smote = SMOTE(random_state=42)
x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)

# 2) Modelleri tanımla
models = {
    'XGBoost': XGBClassifier(
        objective='multi:softmax', num_class=3, eval_metric='mlogloss',
        learning_rate=0.1, max_depth=5, n_estimators=500,
        subsample=0.8, colsample_bytree=1.0,
        n_jobs=-1, random_state=42
    ),
    'LightGBM': LGBMClassifier(
        learning_rate=0.05,
        n_estimators=500,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=1.0,

        # yeni ekleyeceğiniz parametreler
        min_split_gain=0.1,  # split yapmanın minimum kazancı
        min_child_samples=20,  # bir yaprakta kalabilmesi için gereken minimum örnek sayısı

        # log bastırmamak için
        verbose=-1,

        # paralel çalıştırma
        n_jobs=-1,
        random_state=42
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=5,
        random_state=42,

        logging_level='Silent'
    )
}

# 3) Eğit & değerlendir
results = {}
for name, model in models.items():
    model.fit(x_train_sm, y_train_sm)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    results[name] = (acc, f1)

# 4) Karşılaştır
for name, (acc, f1) in results.items():
    print(f"{name:8s} — Accuracy: {acc:.3f}, F1-macro: {f1:.3f}")
    print(classification_report(y_test, models[name].predict(x_test)))
    print("Confusion Matrix:\n", confusion_matrix(y_test, models[name].predict(x_test)), "\n")
