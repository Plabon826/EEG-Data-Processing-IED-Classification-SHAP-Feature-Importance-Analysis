# ======================================================
#  Lever-2  (gamma & min_child_weight) – full workflow
#         with 4-way accuracy & SMOTE
# ======================================================

# ---------- installs (uncomment if first time) ----------
# !pip install -q numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost

# ---------- Imports ----------
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------- Data ----------
DATA_PATH = '/content/merged_all_5.csv'   # ✏️ adjust if needed
df = pd.read_csv(DATA_PATH).dropna()
df = df.drop(columns=['File_Epoch']) if 'File_Epoch' in df.columns else df

X = df.drop(columns=['Label'])
y = df['Label'] - df['Label'].min()          # 0-index labels
NUM_CLASSES = y.nunique()

# ---------- 80 / 10 / 10 split (stratified) ----------
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
)

# ---------- SMOTE on *training* only ----------
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# ---------- Model (Lever-2 regs: gamma & min_child_weight) ----------
model = XGBClassifier(
    objective='multi:softmax',      # returns class labels directly
    num_class=NUM_CLASSES,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42,
    n_jobs=-1,

    # capacity
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,

    # Lever-2 regularisation
    gamma=5.0,               # big gain needed to split
    min_child_weight=20,     # block tiny leaves

    subsample=0.8,
    colsample_bytree=0.8,
)

# Train on the *balanced* data
model.fit(X_train_res, y_train_res,
          eval_set=[(X_val, y_val)],
          verbose=False)

# ---------- 4-way reporter ----------
def full_report(model, label='Model'):
    splits = {
        'Train-SMOTE': (X_train_res, y_train_res),
        'Train-ORIG ': (X_train,     y_train),
        'Val        ': (X_val,       y_val),
        'Test       ': (X_test,      y_test)
    }
    for tag, (X_, y_) in splits.items():
        preds = model.predict(X_)
        acc   = accuracy_score(y_, preds)
        print(f'\n📊 {label} | {tag} accuracy: {acc:.4f}')
        print(classification_report(y_, preds, digits=3, zero_division=0))

        # Confusion matrix for every split
        cm = confusion_matrix(y_, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm,
                    annot=True, fmt='d', cmap='coolwarm',
                    cbar=False, square=True,
                    xticklabels=range(NUM_CLASSES),
                    yticklabels=range(NUM_CLASSES))
        plt.title(f'{label} – {tag.strip()} Confusion Matrix')
        plt.xlabel('Predicted label'); plt.ylabel('True label')
        plt.tight_layout()
        plt.show()

# ---------- Go! ----------
full_report(model, 'xgboost  ')