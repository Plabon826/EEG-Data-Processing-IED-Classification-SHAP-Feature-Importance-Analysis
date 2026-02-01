# ======================================================
#   CatBoost · Lever-2  +  SMOTE  +  4-way reporting
# ======================================================
#
#   ▸ Repro-friendly: fixed random seeds
#   ▸ Handles   – Train-SMOTE / Train-ORIG / Val / Test
#   ▸ Spits out – Accuracy, full classification report,
#                 and a confusion-matrix heat-map for
#                 EVERY split (so you spot trouble fast)
# ------------------------------------------------------

# ---------- installs (uncomment the next line if needed) ----------
# !pip install -q numpy pandas matplotlib seaborn scikit-learn imbalanced-learn catboost

# ---------- imports ----------
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

# ---------- data ----------
DATA_PATH = '/content/merged_all_5.csv'   # ← tweak if you’re elsewhere
df = pd.read_csv(DATA_PATH).dropna()

# optional “house-keeping” column
df = df.drop(columns=['File_Epoch']) if 'File_Epoch' in df.columns else df

# labels to zero-origin (CatBoost likes that)
X = df.drop(columns=['Label'])
y = df['Label'] - df['Label'].min()
NUM_CLASSES = y.nunique()

# ---------- 80 / 10 / 10 split (stratified) ----------
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

# ---------- SMOTE on *training* only ----------
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# ---------- CatBoost (Lever-2-ish params) ----------
cat = CatBoostClassifier(
    loss_function='MultiClass',
    iterations=300,
    depth=4,
    learning_rate=0.1,
    l2_leaf_reg=10.0,          # hefty L2 keeps leaves honest
    min_data_in_leaf=20,       # blocks tiny leaves
    random_strength=1.0,       # split noise (≈ gamma)
    bootstrap_type='Bernoulli',
    subsample=0.8,
    rsm=0.8,                   # feature sampling
    random_seed=42,
    verbose=False
)

cat.fit(
    X_train_res, y_train_res,
    eval_set=(X_val, y_val),
    use_best_model=False      # keep the full 300 iters; flip to True for early-stop
)

# ---------- universal reporter ----------
def full_report(model):
    """Accuracy, classification report & confusion matrix for each split."""
    sets = {
        'Train-SMOTE': (X_train_res, y_train_res),
        'Train-ORIG ': (X_train,     y_train),
        'Val        ': (X_val,       y_val),
        'Test       ': (X_test,      y_test)
    }
    for tag, (X_set, y_set) in sets.items():
        preds = model.predict(X_set).astype(int).ravel()
        acc   = accuracy_score(y_set, preds)
        print(f'\n📊 CatBoost | {tag} accuracy: {acc:.4f}')
        print(classification_report(y_set, preds, digits=3, zero_division=0))
        cm = confusion_matrix(y_set, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm,
                    annot=True, fmt='d', cmap='coolwarm',
                    cbar=False, square=True,
                    xticklabels=range(NUM_CLASSES),
                    yticklabels=range(NUM_CLASSES))
        plt.title(f'CatBoost – {tag.strip()} Confusion Matrix')
        plt.xlabel('Predicted label'); plt.ylabel('True label')
        plt.tight_layout()
        plt.show()

# ---------- go! ----------
full_report(cat)