# ============================================================
# PIPELINE KTH 4 MODELOS + VOTING + TEMPO DE EXECUÇÃO
# Avaliação Exclusiva no Conjunto de Teste Final
# Features: HOF
# Modelos:
#   - Logistic Regression
#   - SVM RBF
#   - AdaBoost
#   - XGBoost
# ============================================================

import os
import sys
import random
import itertools
import hashlib
from pathlib import Path
import time
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# 1. CONFIGURAÇÃO DE DETERMINISMO, DIRETÓRIOS E VARIÁVEIS GLOBAIS
# ============================================================

SEED = 42
HYPER_RESULTS_DIR = None  # Se None, busca automaticamente a pasta resultados_hyperparametros_*
WEIGHT_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]  # Valores para busca de pesos do ensemble

# Diretório de Saída
OUTPUT_DIR = Path("resultados_modelos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 2. CLASSES (KTH)
# ============================================================

CLASS_MAP = {
    "walking": 0,
    "jogging": 1,
    "running": 2,
    "boxing": 3,
    "handclapping": 4,
    "handwaving": 5,
}

N_CLASSES = len(CLASS_MAP)

# ============================================================
# 3. CARREGAR FEATURES (KTH .npy) E DIVISÃO DE TESTE
# ============================================================

FEATURE_FOLDER = "hof_features_kth"
print("Carregando features...")

X_list = []
y_list = []
persons_list = []

for filename in os.listdir(FEATURE_FOLDER):
    if not filename.endswith(".npy"):
        continue

    # Exemplo de arquivo: person01_boxing_d1_1_95.npy
    parts = filename.split("_")
    person = parts[0]
    action = parts[1]

    features = np.load(os.path.join(FEATURE_FOLDER, filename))

    X_list.append(features)
    y_list.append(CLASS_MAP[action])
    persons_list.append(person)

# Converter para NumPy
X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
persons = np.array(persons_list)

print("Informações do dataset:")
print("X:", X.shape)
print("y:", y.shape)
print("Pessoas únicas:", len(np.unique(persons)))
print("Sequências:", len(X))
print("Classes mapeadas:", list(CLASS_MAP.keys()))

# Localizar diretório com os hiperparâmetros e divisões salvas
if HYPER_RESULTS_DIR is None:
    candidates = sorted(Path(".").glob("resultados_hyperparametros_*/divisao_indices.npz"))
    if not candidates:
        raise RuntimeError("Nenhuma pasta 'resultados_hyperparametros_*' contendo 'divisao_indices.npz' foi encontrada.")
    hyper_dir = candidates[-1].parent  # Pega o diretório mais recente
else:
    hyper_dir = Path(HYPER_RESULTS_DIR)

print(f"\nCarregando divisão e configurações de: {hyper_dir}")

# Validação do Hash do Dataset
hash_file = hyper_dir / "dataset_sha256.txt"
if hash_file.exists():
    expected_hash = hash_file.read_text().strip()
    actual_hash = hashlib.sha256()
    for array in (X, y, persons.astype("U")):
        actual_hash.update(np.ascontiguousarray(array).tobytes())
    if actual_hash.hexdigest() != expected_hash:
        raise RuntimeError("O arquivo de features ou sua ordem difere daquela usada no script de hiperparâmetros!")

# Carregar Índices da Divisão
with np.load(hyper_dir / "divisao_indices.npz") as division:
    development_idx = division["development_idx"]
    final_test_idx = division["final_test_idx"]

# Sanity Check de Integridade
if (len(np.unique(np.r_[development_idx, final_test_idx])) != len(X)
        or len(development_idx) + len(final_test_idx) != len(X)
        or set(persons[development_idx]) & set(persons[final_test_idx])):
    raise RuntimeError("Divisão inválida ou com vazamento de pessoas entre desenvolvimento e teste!")

selected_df = pd.read_csv(hyper_dir / "melhores_configuracoes.csv").set_index("modelo")

# ============================================================
# 4. CRIAR MODELOS COM HIPERPARÂMETROS OTIMIZADOS
# ============================================================

def create_models():
    logistic_params = selected_df.loc["logistic"]
    svm_params = selected_df.loc["svm"]
    adaboost_params = selected_df.loc["adaboost"]
    xgb_params = selected_df.loc["xgboost"]

    logistic = LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs",
        max_iter=1000, random_state=SEED
    )
    svm = SVC(
        kernel="rbf", C=1.0,
        gamma="scale", degree=2, probability=True, random_state=SEED
    )
    adaboost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=6, random_state=SEED
        ),
        n_estimators=200,
        learning_rate=1.0, random_state=SEED
    )
    xgboost = XGBClassifier(
        objective="multi:softprob", num_class=N_CLASSES,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED, eval_metric="mlogloss", n_jobs=1,
        tree_method="hist"
    )
    return {"logistic": logistic, "svm": svm, "adaboost": adaboost, "xgboost": xgboost}

# ============================================================
# 5. FUNÇÕES AUXILIARES DE MÉTRICAS E AVALIAÇÃO
# ============================================================

def get_probabilities(models, X_data):
    return {name: model.predict_proba(X_data) for name, model in models.items()}

def calculate_metrics(y_true, y_pred, probabilities):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
    except ValueError:
        auc = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def save_confusion_matrix(y_true, y_pred, method_name, fold):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES))
    row_sums = cm.sum(axis=1, keepdims=True)

    cm_normalized = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
    ) * 100

    labels = list(CLASS_MAP.keys())
    df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    filename = OUTPUT_DIR / f"matriz_confusao_{method_name}_fold_{fold}_{timestamp}_final.csv"
    df.to_csv(filename)
    return df

def majority_voting(probabilities):
    predictions = []
    for model_name in ["logistic", "svm", "adaboost", "xgboost"]:
        pred = np.argmax(probabilities[model_name], axis=1)
        predictions.append(pred)

    predictions = np.array(predictions)
    final_predictions = []

    for sample_idx in range(predictions.shape[1]):
        votes = predictions[:, sample_idx]
        counts = np.bincount(votes, minlength=N_CLASSES)
        final_predictions.append(np.argmax(counts))

    return np.array(final_predictions)

def weighted_voting(probabilities, weights):
    weighted_probabilities = (
        weights[0] * probabilities["logistic"] +
        weights[1] * probabilities["svm"] +
        weights[2] * probabilities["adaboost"] +
        weights[3] * probabilities["xgboost"]
    )
    weighted_probabilities /= sum(weights)
    predictions = np.argmax(weighted_probabilities, axis=1)
    return predictions, weighted_probabilities

def train_models(X_train, y_train):
    models = create_models()
    training_times = {}

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        training_times[name] = elapsed
        print(f"    {name}: {elapsed:.2f} s")

    return models, training_times

def select_weights(X_outer_train, y_outer_train, persons_outer_train):
    print("\n  Selecionando pesos com validação interna (K-Fold no desenvolvimento)...")

    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    weight_combinations = [w for w in itertools.product(WEIGHT_VALUES, repeat=4) if sum(w) > 0]

    print(f"  Combinações de pesos testadas: {len(weight_combinations)}")

    scores = {weights: [] for weights in weight_combinations}
    inner_fold_people = []

    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
        inner_cv.split(X_outer_train, y_outer_train, groups=persons_outer_train), start=1
    ):
        print(f"    Inner fold {inner_fold}/5")

        X_inner_train = X_outer_train[inner_train_idx]
        X_inner_val = X_outer_train[inner_val_idx]
        y_inner_train = y_outer_train[inner_train_idx]
        y_inner_val = y_outer_train[inner_val_idx]
        persons_inner_train = persons_outer_train[inner_train_idx]
        persons_inner_val = persons_outer_train[inner_val_idx]

        overlap = set(persons_inner_train).intersection(set(persons_inner_val))
        if overlap:
            raise RuntimeError(f"LEAKAGE DETECTADO na validação interna: {overlap}")

        inner_fold_people.append({
            "fold": inner_fold,
            "train_people": sorted(np.unique(persons_inner_train)),
            "val_people": sorted(np.unique(persons_inner_val))
        })

        scaler = StandardScaler()
        X_inner_train_scaled = scaler.fit_transform(X_inner_train)
        X_inner_val_scaled = scaler.transform(X_inner_val)

        models, _ = train_models(X_inner_train_scaled, y_inner_train)
        probabilities = get_probabilities(models, X_inner_val_scaled)

        for weights in weight_combinations:
            pred, _ = weighted_voting(probabilities, weights)
            score = f1_score(y_inner_val, pred, average="macro", zero_division=0)
            scores[weights].append(score)

    mean_scores = {weights: np.mean(values) for weights, values in scores.items()}
    best_weights = max(mean_scores, key=mean_scores.get)
    best_score = mean_scores[best_weights]

    print(f"\n  Melhores pesos selecionados: {best_weights}")
    print(f"  F1-Score médio interno: {best_score:.4f}")

    return best_weights, inner_fold_people

# ============================================================
# 6. TREINAMENTO EM DEV E AVALIAÇÃO EM TESTE FINAL
# ============================================================

total_start = time.perf_counter()

all_results = []
all_confusion_matrices = {
    "logistic": [],
    "svm": [],
    "adaboost": [],
    "xgboost": [],
    "voting": [],
    "weighted_voting": []
}

selected_weights = []
external_fold_people = []
all_inner_fold_people = []
total_model_training_time = {"logistic": 0.0, "svm": 0.0, "adaboost": 0.0, "xgboost": 0.0}

for fold, (train_idx, test_idx) in enumerate([(development_idx, final_test_idx)], start=1):
    fold_start = time.perf_counter()

    print("\n" + "=" * 70)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE FINAL")
    print("=" * 70)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    persons_train, persons_test = persons[train_idx], persons[test_idx]

    overlap = set(persons_train).intersection(set(persons_test))
    if overlap:
        raise RuntimeError(f"LEAKAGE DETECTADO no conjunto de teste: {overlap}")

    print("\nPessoas Treino (Desenvolvimento):", sorted(np.unique(persons_train)))
    print("Pessoas Teste (Teste Final):", sorted(np.unique(persons_test)))
    print(f"Qtd. Pessoas Treino: {len(np.unique(persons_train))} | Qtd. Pessoas Teste: {len(np.unique(persons_test))}")

    external_fold_people.append({
        "fold": fold,
        "train_people": ",".join(sorted(np.unique(persons_train))),
        "test_people": ",".join(sorted(np.unique(persons_test)))
    })

    # Seleção dos pesos usando apenas os dados de desenvolvimento
    best_weights, inner_people = select_weights(X_train, y_train, persons_train)

    selected_weights.append({
        "fold": fold,
        "logistic_weight": best_weights[0],
        "svm_weight": best_weights[1],
        "adaboost_weight": best_weights[2],
        "xgboost_weight": best_weights[3]
    })

    all_inner_fold_people.append({"outer_fold": fold, "inner_folds": inner_people})

    # Escalamento (Scaler fit apenas no desenvolvimento)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treino dos modelos finais em todo o desenvolvimento
    print("\nTreinando modelos finais no conjunto de Desenvolvimento completo...")
    models, training_times = train_models(X_train_scaled, y_train)

    for name, elapsed in training_times.items():
        total_model_training_time[name] += elapsed

    # Predição no conjunto de Teste Final
    probabilities = get_probabilities(models, X_test_scaled)

    # Avaliação Individual
    for model_name in ["logistic", "svm", "adaboost", "xgboost"]:
        y_pred = np.argmax(probabilities[model_name], axis=1)
        metrics = calculate_metrics(y_test, y_pred, probabilities[model_name])

        all_results.append({
            "fold": fold,
            "method": model_name,
            **metrics
        })

        cm = save_confusion_matrix(y_test, y_pred, model_name, fold)
        all_confusion_matrices[model_name].append(cm)

    # Majority Voting
    voting_pred = majority_voting(probabilities)
    voting_probabilities = (
        probabilities["logistic"] + probabilities["svm"] +
        probabilities["adaboost"] + probabilities["xgboost"]
    ) / 4.0
    voting_metrics = calculate_metrics(y_test, voting_pred, voting_probabilities)

    all_results.append({"fold": fold, "method": "voting", **voting_metrics})
    cm = save_confusion_matrix(y_test, voting_pred, "voting", fold)
    all_confusion_matrices["voting"].append(cm)

    # Weighted Voting
    weighted_pred, weighted_probabilities = weighted_voting(probabilities, best_weights)
    weighted_metrics = calculate_metrics(y_test, weighted_pred, weighted_probabilities)

    all_results.append({"fold": fold, "method": "weighted_voting", **weighted_metrics})
    cm = save_confusion_matrix(y_test, weighted_pred, "weighted_voting", fold)
    all_confusion_matrices["weighted_voting"].append(cm)

    fold_time = time.perf_counter() - fold_start
    print(f"\nTempo total da avaliação: {fold_time:.2f} s")

# ============================================================
# 7. EXIBIÇÃO E SALVAMENTO DOS RESULTADOS (DENTRO DE RESULTADOS_MODELOS)
# ============================================================

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 70)
print("RESULTADOS NO CONJUNTO DE TESTE FINAL")
print("=" * 70)
print(results_df.to_string(index=False))

summary_df = results_df.groupby("method")[["accuracy", "precision", "recall", "f1", "auc"]].agg(["mean"])

print("\n" + "=" * 70)
print("RESUMO DAS MÉTRICAS")
print("=" * 70)
print(summary_df)

# Salvamento de relatórios CSV dentro da pasta OUTPUT_DIR (resultados_modelos)
pd.DataFrame(external_fold_people).to_csv(OUTPUT_DIR / f"pessoas_folds_externos_{timestamp}_final.csv", index=False)

inner_rows = []
for outer_data in all_inner_fold_people:
    for inner_data in outer_data["inner_folds"]:
        inner_rows.append({
            "outer_fold": outer_data["outer_fold"],
            "inner_fold": inner_data["fold"],
            "train_people": ",".join(inner_data["train_people"]),
            "validation_people": ",".join(inner_data["val_people"])
        })

pd.DataFrame(inner_rows).to_csv(OUTPUT_DIR / f"pessoas_folds_internos_{timestamp}_final.csv", index=False)
results_df.to_csv(OUTPUT_DIR / f"resultados_por_fold_{timestamp}_final.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / f"resumo_resultados_{timestamp}_final.csv")
pd.DataFrame(selected_weights).to_csv(OUTPUT_DIR / f"pesos_selecionados_{timestamp}_final.csv", index=False)

for method_name, matrices in all_confusion_matrices.items():
    if matrices:
        matrices[0].to_csv(OUTPUT_DIR / f"matriz_confusao_teste_{method_name}_{timestamp}_final.csv")

total_time = time.perf_counter() - total_start

pd.DataFrame({
    "modelo": list(total_model_training_time.keys()),
    "tempo_total_treinamento_segundos": list(total_model_training_time.values())
}).to_csv(OUTPUT_DIR / f"tempo_treinamento_modelos_{timestamp}_final.csv", index=False)

pd.DataFrame({"tempo_total_execucao_segundos": [total_time]}).to_csv(
    OUTPUT_DIR / f"tempo_total_execucao_{timestamp}_final.csv", index=False
)

print("\n" + "=" * 70)
print("EXECUÇÃO FINALIZADA COM SUCESSO")
print("=" * 70)
print(f"Timestamp: {timestamp}")
print(f"Resultados salvos em: {OUTPUT_DIR.resolve()}")
print(f"Tempo total: {total_time:.2f} segundos")