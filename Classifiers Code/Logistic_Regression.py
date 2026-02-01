# ======================================================
#  Multinomial Logistic Regression – full workflow
#             with SMOTE & 4-way reporting
# ======================================================

# ---------- installs (uncomment on first run) ----------
# !pip install -q numpy pandas matplotlib seaborn scikit-learn imbalanced-learn

# ---------- Imports ----------
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# ---------- Data ----------
DATA_PATH = '/content/merged_all_5.csv'   # ✏️ adjust if needed
df = pd.read_csv(DATA_PATH).dropna()
df = df.drop(columns=['File_Epoch']) if 'File_Epoch' in df.columns else df

X = df.drop(columns=['Label'])
y = df['Label'] - df['Label'].min()       # zero-origin labels
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

# ---------- Logistic-Regression model ----------
logreg = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',           # supports multinomial
    multi_class='multinomial',
    max_iter=1000,
    random_state=42
)

# Train on the balanced data
logreg.fit(X_train_res, y_train_res)

# ---------- 4-way reporter ----------
def full_report(model, label='Model'):
    datasets = {
        'Train-SMOTE': (X_train_res, y_train_res),
        'Train-ORIG ': (X_train,     y_train),
        'Val        ': (X_val,       y_val),
        'Test       ': (X_test,      y_test)
    }
    for tag, (X_, y_) in datasets.items():
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
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

# ---------- Go! ----------
full_report(logreg, 'Logistic Regression (multinomial)')