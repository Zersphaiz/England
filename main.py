import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_curve, \
    roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

def run_logistic(x_train, y_train, x_test, y_test):
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
    return model


# ---------------------------------------------
# Random Forest Classifier
# ---------------------------------------------

def run_random_forest(x_train, y_train, x_test, y_test):
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

    return rf_model


# ---------------------------------------------
# XGBoost
# ---------------------------------------------

def run_ensemble(x_train, y_train, x_test, y_test):
    smote = SMOTE(random_state=42)
    x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)

    xgb = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        learning_rate=0.1,
        max_depth=5,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1
    )

    lgbm = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    cat = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=5,
        loss_function='MultiClass',
        random_seed=42,
        verbose=0
    )

    # --- 2) VotingClassifier ile ensemble ---

    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
        voting='soft',  # Olasılıkları toplayıp en yüksek olasılığa sahip sınıfı seç
        weights=[1, 1, 1],  # İsterseniz model ağırlıklarını ayarlayabilirsiniz
        n_jobs=-1
    )

    # --- 3) Eğit ve Tahmin ---

    # Eğer SMOTE ile dengelenmiş veriniz varsa:
    X_train_fit, y_train_fit = x_train_sm, y_train_sm
    # Yoksa normal:
    #   X_train_fit, y_train_fit = x_train, y_train

    ensemble.fit(X_train_fit, y_train_fit)
    y_pred_ens = ensemble.predict(x_test)

    # --- 4) Performans Raporu ---
    print("Ensemble Accuracy:", accuracy_score(y_test, y_pred_ens))
    print("\nEnsemble Classification Report:\n", classification_report(y_test, y_pred_ens))
    print("\nEnsemble Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ens))

    return ensemble


def run_two_stage(x_train, y_train, x_test, y_test, recall_target=0.30):
    # --- AŞAMA 1: Beraberlik vs. Beraberlik Dışı (binary) ---
    # Hedef değişkeni 1: beraberlik, 0: non-draw
    y_train_draw = (y_train == 0).astype(int)
    y_test_draw = (y_test == 0).astype(int)

    # SMOTE burada da kullanabiliriz
    sm = SMOTE(random_state=42)
    X_tr_draw, y_tr_draw = sm.fit_resample(x_train, y_train_draw)

    # Basit bir XGBoost ikili sınıflandırıcı
    clf_draw = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        learning_rate=0.1,
        max_depth=3,
        n_estimators=300,
        n_jobs=-1,
        random_state=42
    )
    clf_draw.fit(X_tr_draw, y_tr_draw)

    # --- AŞAMA 2: “Non-draw”lar için Home/Away Çoklu Sınıflandırma ---
    mask_nd = (y_train != 0)
    X_tr_nd = x_train[mask_nd]
    y_tr_nd_bin = (y_train[mask_nd] == 1).astype(int)

    clf_nd = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        learning_rate=0.1,
        max_depth=5,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=1.0,
        n_jobs=-1,
        random_state=42
    )
    clf_nd.fit(X_tr_nd, y_tr_nd_bin)
    # 1) Draw olasılıklarını yeniden al
    proba_draw = clf_draw.predict_proba(x_test)[:, 1]
    y_test_draw = (y_test == 0).astype(int)

    # 2) %30 recall₀ için eşik belirle
    pos_probs = proba_draw[y_test_draw == 1]
    best_thr = np.percentile(pos_probs, 100 * (1 - 0.30))
    pred_draw = (proba_draw >= best_thr).astype(int)

    # 3) final_pred’i oluştur
    final_pred = np.zeros_like(y_test, dtype=int)
    final_pred[pred_draw == 1] = 0
    idx_nd = np.where(pred_draw == 0)[0]
    bin_preds = clf_nd.predict(x_test.iloc[idx_nd])
    final_pred[idx_nd] = np.where(bin_preds == 1, 1, 2)

    # 4) Metrikleri yazdır
    print("✅ Two-Stage (Threshold Tuned) Final Results:")
    print("Accuracy:", accuracy_score(y_test, final_pred))
    print("\nClassification Report:\n", classification_report(y_test, final_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, final_pred))
    print("Recall (class=0):", recall_score(y_test_draw, pred_draw))

    return clf_draw, clf_nd


# xgb en başarılı eski blok

def run_xgb_grid_and_threshold(x_train, y_train, x_test, y_test):
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
        'colsample_bytree': [0.8, 1.0],
        'learning_rate': [0.1],  # sabit
        'n_estimators': [500]  # sabit
    }

    # Grid Search başlat
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='accuracy',
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

    ###aşağı kısım deneme

    # 1) Olasılıkları al
    y_proba = best_model.predict_proba(x_test)
    y_test_draw = (y_test == 0).astype(int)

    # 2) Precision–Recall eğrisinden eşik ve recall’ları al
    prec, rec, thr = precision_recall_curve(y_test_draw, y_proba[:, 0])

    # 3) En iyi eşiği bul: recall >= 0.30 koşulunu sağlayanlar içinde en yüksek accuracy
    best_acc = 0
    best_thr = 0.5  # default
    best_rec = 0

    for t, r in zip(thr, rec[1:]):  # rec[1:] çünkü thr uzunluğu rec-1
        # Draw / non-draw tahmini
        is_draw = (y_proba[:, 0] >= t).astype(int)
        # İkili olmayanlar için argmax(1:) + 1
        pred = np.where(is_draw == 1, 0, np.argmax(y_proba[:, 1:], axis=1) + 1)
        acc = accuracy_score(y_test, pred)
        if r >= 0.30 and acc > best_acc:
            best_acc = acc
            best_thr = t
            best_rec = r

    print(f"✔ Seçilen eşik: {best_thr:.3f}  →  Accuracy={best_acc:.3f}, Recall₀={best_rec:.3f}")

    # 4) Son metrikleri yazdır
    final_pred = np.where(y_proba[:, 0] >= best_thr, 0, np.argmax(y_proba[:, 1:], axis=1) + 1)
    print("\n👉 XGBoost (Grid + Opt Thr) Sonuçları:")
    print("Accuracy:", accuracy_score(y_test, final_pred))
    print("Draw Recall:", recall_score(y_test_draw, (final_pred == 0).astype(int)))
    print("\nClassification Report:\n", classification_report(y_test, final_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, final_pred))

    return best_acc, best_thr


# final_xgb = XGBClassifier(
#     objective='multi:softprob',  # softmax yerine softprob, çünkü predict_proba kullanacağız
#     num_class=3,
#     eval_metric='mlogloss',
#     learning_rate=0.1,
#     max_depth=5,
#     n_estimators=500,
#     subsample=0.8,
#     colsample_bytree=1.0,
#     n_jobs=-1,
#     random_state=42
# )
# final_xgb.fit(x_train_sm, y_train_sm)
#
# # Tahmin olasılıklarını al
# y_proba = final_xgb.predict_proba(x_test)
#
# # Örneğin 3 sınıf için özel eşikler (bunlar örnek değerlerdir, deneyerek ayarlayacağız)
# thresholds = [0.3, 0.4, 0.3]  # [class_0, class_1, class_2]
#
# # Olasılıkları eşiklere göre değerlendirerek sınıf seç
# adjusted_preds = []
# for probs in y_proba:
#     adjusted = [p / t for p, t in zip(probs, thresholds)]
#     adjusted_preds.append(np.argmax(adjusted))
#
# # Performansı değerlendirme
# print("Threshold-Tuned XGBoost Accuracy:", accuracy_score(y_test, adjusted_preds))
# print("\nClassification Report:\n", classification_report(y_test, adjusted_preds))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, adjusted_preds))
#
# # --- 1) Modeli (örnek: XGBoost) olasılık çıktısıyla eğit ---
# xgb = XGBClassifier(
#     objective='multi:softprob',  # softprob = olasılık döndür
#     num_class=3,
#     learning_rate=0.1,
#     max_depth=5,
#     n_estimators=500,
#     subsample=0.8,
#     colsample_bytree=1.0,
#     random_state=42,
#     n_jobs=-1
# )
# xgb.fit(x_train_sm, y_train_sm)
#
# # (İsteğe bağlı) çıktı kalibrasyonu → olasılıkları daha güvenilir kılar
# xgb = CalibratedClassifierCV(xgb, method='isotonic', cv=3).fit(x_train_sm, y_train_sm)
#
# # --- 2) Test kümesi olasılıkları ---
# y_proba = xgb.predict_proba(x_test)
#
# proba = xgb.predict_proba(x_test)
#
# for cls in [0,1,2]:
#     # ikili etiket
#     y_true = (y_test == cls).astype(int)
#
#     # Precision–Recall eğrisi
#     prec, rec, thr_pr = precision_recall_curve(y_true, proba[:, cls])
#     f1_scores = 2 * prec * rec / (prec + rec + 1e-12)
#     best_idx = f1_scores.argmax()
#     best_thr = thr_pr[best_idx]
#
#     plt.figure()
#     plt.plot(rec, prec, label=f"Class {cls} (best F1={f1_scores[best_idx]:.2f} @thr={best_thr:.2f})")
#     plt.scatter(rec[best_idx], prec[best_idx], c='red')
#     plt.title(f"PR Curve – Class {cls}")
#     plt.xlabel("Recall"); plt.ylabel("Precision")
#     plt.legend()
#     plt.show()
#
#     # ROC eğrisi
#     fpr, tpr, thr_roc = roc_curve(y_true, proba[:, cls])
#     plt.figure()
#     plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.2f}")
#     plt.plot([0,1],[0,1], 'k--')
#     plt.title(f"ROC Curve – Class {cls}")
#     plt.xlabel("FPR"); plt.ylabel("TPR")
#     plt.legend()
#     plt.show()
#
# print("Bu grafikleri inceleyip, beraberlik (class=0) için recall’u ~0.3’e çıkartacak eşikleri seçiyoruz.")
#
# # --- (1) Kalibre edilmiş modelden olasılıkları alınmış olsun zaten: y_proba = xgb.predict_proba(x_test) ---
#
# # --- (2) Beraberlik (class=0) için precision–recall eğrisini çıkar
# y_true_0 = (y_test == 0).astype(int)
# precisions, recalls, thresholds = precision_recall_curve(y_true_0, y_proba[:, 0])
#
# # --- (3) recall >= %30 sağlayan ilk eşiği bul
# target_recall = 0.30
# idxs = np.where(recalls >= target_recall)[0]
# if len(idxs) > 0:
#     thr0 = thresholds[idxs[0]]
# else:
#     thr0 = thresholds[np.argmax(recalls)]
#
# # --- (4) Diğer sınıflar için de eşik belirleyin (isterseniz eşit dağılım da yapabilirsiniz)
# best_thresh = {
#     0: thr0,
#     1: 1/3,   # ya da önceki grid ile bulunmuş eşiklerinizden biri
#     2: 1/3
# }
#
# print("Seçilen eşikler (class=0 için recall>=0.30):", best_thresh)
#
# # --- (5) Tahmin fonksiyonunuz
# def predict_with_thresholds(proba, thr_dict):
#     preds = []
#     for p in proba:
#         passed = [c for c,t in thr_dict.items() if p[c] >= t]
#         if passed:
#             preds.append(max(passed, key=lambda c: p[c]))
#         else:
#             preds.append(np.argmax(p))
#     return np.array(preds)
#
# # --- (6) Eşikli tahmin
# y_pred_thr = predict_with_thresholds(y_proba, best_thresh)
#
# # --- (7) Sonuçları yazdır
# print("Threshold-Tuned Accuracy:", accuracy_score(y_test, y_pred_thr))
# print("\nClassification Report:\n", classification_report(y_test, y_pred_thr))
# print("Recall (class=0):", recall_score((y_test==0).astype(int),
#                                         (y_pred_thr==0).astype(int)))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_thr))
#

# ---------------------------------------------
# kNN
# ---------------------------------------------
def run_knn(x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    sm_knn = SMOTE(random_state=42)
    X_train_knn_sm, y_train_knn_sm = sm_knn.fit_resample(X_train_scaled, y_train)
    print("SMOTE sonrası sınıf dağılımı:", np.bincount(y_train_knn_sm))

    knn = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [9],
        'weights': ['distance'],
        'p': [1]
    }

    grid_knn = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    # veriyi kullan
    grid_knn.fit(X_train_knn_sm, y_train_knn_sm)

    print("Best KNN params:", grid_knn.best_params_)

    best_knn = grid_knn.best_estimator_
    y_pred_knn = best_knn.predict(X_test_scaled)

    print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn))
    print(confusion_matrix(y_test, y_pred_knn))

    draw_recall = recall_score((y_test == 0).astype(int), (y_pred_knn == 0).astype(int))
    print("KNN Draw Recall:", draw_recall)

    return best_knn


# ---------------------------------------------
# MLPClassifier
# ---------------------------------------------

def run_mlp(x_train_sm, y_train_sm, x_test, y_test, recall_target=0.30):
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        max_iter=500,
        random_state=42
    )
    mlp.fit(x_train_sm, y_train_sm)

    # --- A) baseline metrikler ---
    y_pred_mlp = mlp.predict(x_test)
    print("MLP — Accuracy:", accuracy_score(y_test, y_pred_mlp))
    print("MLP — F1-macro:", f1_score(y_test, y_pred_mlp, average='macro'))
    print(classification_report(y_test, y_pred_mlp))
    print("MLP Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp))

    # --- B) class=0 için en iyi eşik değerini precision‐recall eğrisi üzerinden bul ---
    from sklearn.metrics import precision_recall_curve, recall_score

    probs0 = mlp.predict_proba(x_test)[:, 0]
    prec, rec, thr = precision_recall_curve((y_test == 0).astype(int), probs0)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-12)

    mask = rec >= 0.30
    if mask.any():
        best_i = np.argmax(f1_scores[mask])
        best_thr = thr[np.where(mask)[0][best_i]]
    else:
        best_i = np.argmax(f1_scores)
        best_thr = thr[best_i]

    print(f"MLP class=0 için en iyi eşik: {best_thr:.3f}, Recall: {rec[best_i]:.3f}, F1: {f1_scores[best_i]:.3f}")

    # --- C) Bu eşiğe göre tahminleri ayarla ve metrikleri tekrar yazdır ---
    adjusted = []
    for p in mlp.predict_proba(x_test):
        adjusted.append(0 if p[0] >= best_thr else np.argmax(p))
    y_pred_mlp_thr = np.array(adjusted)

    print("\nThreshold-Tuned MLP:")
    print(" Accuracy:", accuracy_score(y_test, y_pred_mlp_thr))
    y_true_0 = (y_test == 0).astype(int)
    y_pred_0 = (y_pred_mlp_thr == 0).astype(int)
    print("Recall (class=0):", recall_score(y_true_0, y_pred_0))
    print(classification_report(y_test, y_pred_mlp_thr))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp_thr))

    return mlp


# ---------------------------------------------
# Support Vector Machine
# ---------------------------------------------
def run_svm_all(x_train_sm, y_train_sm, x_train, y_train, x_test, y_test):
    sm_svm = SMOTE(random_state=42)
    x_train_sm, y_train_sm = sm_svm.fit_resample(x_train, y_train)

    X_train_fit, y_train_fit = x_train_sm, y_train_sm

    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',  # dengesiz sınıflar için
        probability=True,  # olasılık tabanlı eşik optimizasyonu isterseniz
        random_state=42
    )

    svm_model.fit(X_train_fit, y_train_fit)
    y_pred_svm = svm_model.predict(x_test)

    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))
    print("\nSVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

    # --- 1) Pipeline ile ölçeklendirme + tek SVM ---
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            C=1.0,  # varsayılan değer
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=False,  # predict_proba ihtiyacın yoksa False yap
            random_state=42
        ))
    ])

    # --- 2) Eğit ve değerlendir ---
    svm.fit(x_train, y_train)
    y_pred_svm = svm.predict(x_test)

    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))
    print("\nSVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

    # ---------------------------------------------
    # Yeni Blok: Pipeline + RandomizedSearchCV SVM
    # ---------------------------------------------

    # 1) Ölçeklendirme + SVM pipeline'ı
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=False,  # proba gerek yoksa False
            random_state=42
        ))
    ])

    # 2) Parametre dağılımı
    param_dist = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto']
    }

    # 3) RandomizedSearchCV
    rand_svm = RandomizedSearchCV(
        estimator=svm_pipe,
        param_distributions=param_dist,
        n_iter=6,  # çok ağır olmasın
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    # 4) Eğit
    rand_svm.fit(x_train, y_train)

    # 5) Sonuçları yazdır
    print("En iyi SVM parametreleri:", rand_svm.best_params_)
    y_opt = rand_svm.predict(x_test)
    print("Optimize SVM Accuracy:", accuracy_score(y_test, y_opt))
    print(classification_report(y_test, y_opt))
    print(confusion_matrix(y_test, y_opt))

    # ---------------------------------------------
    #  SVM için Threshold Tuning
    # ---------------------------------------------

    # 1) En iyi parametreli modeli alıp kalibre et
    best_svm = rand_svm.best_estimator_  # RandomizedSearchCV çıktısı
    svm_cal = CalibratedClassifierCV(best_svm, method='isotonic', cv=3)
    svm_cal.fit(x_train, y_train)

    # 2) Test kümesi olasılıklarını al
    y_proba_svm = svm_cal.predict_proba(x_test)

    # 3) Beraberlik (class=0) için en iyi eşik noktası
    y_true_0 = (y_test == 0).astype(int)
    prec, rec, thr = precision_recall_curve(y_true_0, y_proba_svm[:, 0])
    f1_scores = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = f1_scores.argmax()
    best_thresh_0 = thr[best_idx]
    print(
        f"Class 0 için en iyi eşik: {best_thresh_0:.3f}  (F1₀={f1_scores[best_idx]:.3f}, Recall₀={rec[best_idx]:.3f})")

    # 4) Tahminleri threshold’a göre ayarla
    def predict_svm_with_thresh(proba, thresh):
        # eşiği geçenleri 0, diğerlerini en yüksek prob’a göre
        preds = []
        for p in proba:
            if p[0] >= thresh:
                preds.append(0)
            else:
                preds.append(p.argmax())
        return np.array(preds)

    y_pred_svm_thr = predict_svm_with_thresh(y_proba_svm, best_thresh_0)

    # 5) Performans raporu
    print("Threshold-Tuned SVM Accuracy:", accuracy_score(y_test, y_pred_svm_thr))
    print("\nThreshold-Tuned SVM Classification Report:\n",
          classification_report(y_test, y_pred_svm_thr))
    print("Threshold-Tuned SVM Confusion Matrix:\n",
          confusion_matrix(y_test, y_pred_svm_thr))
    print(f"Recall (class=0): {recall_score(y_test, y_pred_svm_thr, labels=[0, 1, 2], average=None)[0]:.3f}")

    best_acc = 0
    best_thr = 0
    best_rec0 = 0

    for thr in np.linspace(0, 1, 101):
        # sınıf=0 eşik uygulaması
        preds = [0 if p[0] >= thr else p.argmax() for p in y_proba_svm]
        rec0 = recall_score(y_test, preds, labels=[0, 1, 2], average=None)[0]
        acc = accuracy_score(y_test, preds)
        # recall₀ en az 0.30 olsun, kalanını da mümkün olduğunca yüksek tut
        if rec0 >= 0.30 and acc > best_acc:
            best_acc, best_rec0, best_thr = acc, rec0, thr

    print(f"Seçilen eşik: {best_thr:.2f}  →  Recall₀={best_rec0:.3f}, Accuracy={best_acc:.3f}")

    # Son durumda değerlendirme
    final_preds = [0 if p[0] >= best_thr else p.argmax() for p in y_proba_svm]
    print("\nFinal Threshold-Tuned SVM Report:")
    print(" Accuracy:", accuracy_score(y_test, final_preds))
    print(classification_report(y_test, final_preds))
    print(" Confusion Matrix:\n", confusion_matrix(y_test, final_preds))

    return svm_model, svm_pipe, rand_svm, svm_cal, final_preds


sm = SMOTE(random_state=42)
x_train_sm, y_train_sm = sm.fit_resample(x_train, y_train)

# Run
lr_model = run_logistic(x_train, y_train, x_test, y_test)
rf_model = run_random_forest(x_train, y_train, x_test, y_test)
ensemble_model = run_two_stage(x_train, y_train, x_test, y_test)
clf_draw, clf_nd = run_two_stage(x_train, y_train, x_test, y_test)
best_xgb_model, xgb_threshold = run_xgb_grid_and_threshold(x_train, y_train, x_test, y_test)
best_knn_model = run_knn(x_train, y_train, x_test, y_test)
mlp_model = run_mlp(x_train_sm, y_train_sm, x_test, y_test)
svm_model, svm_pipe, rand_svm, svm_cal, final_preds = run_svm_all(x_train, y_train, x_train_sm, y_train_sm, x_test, y_test)
