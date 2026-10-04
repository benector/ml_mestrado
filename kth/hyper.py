# ============================================================
# PIPELINE KTH - ANÁLISE DE HIPERPARÂMETROS
# Modelos: regressão logística, SVM, AdaBoost e XGBoost
# Features: HOF (/hof_features)
# Divisão por pessoa: desenvolvimento/teste (semente fixa = 42)
# Busca: StratifiedGroupKFold EXCLUSIVAMENTE em desenvolvimento
# O conjunto de teste NUNCA é utilizado ou avaliado neste script.
# ============================================================

import os

SEED = 42

# Determinismo: definir antes dos imports numéricos
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURAÇÕES GERAIS
# ============================================================

FEATURE_FILE = "resnet_features.npz"

N_SPLITS = 5
TEST_SIZE = 0.20


# ============================================================
# 2. HIPERPARÂMETROS
# ============================================================

LOGISTIC_GRID = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "max_iter": [1000],
}

SVM_GRID = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [1.0],
    "gamma": ["scale"],
    "degree": [2],
}

ADABOOST_GRID = {
    "n_estimators": [50, 200],
    "learning_rate": [0.1, 0.5, 1.0],
    "max_depth": [3, 6],
}

XGBOOST_GRID = {
    "n_estimators": [50, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.1, 0.5, 1.0],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


# ============================================================
# 3. TIMESTAMP / DIRETÓRIO DE SAÍDA
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = Path(
    f"resultados_hyperparametros_dev_{timestamp}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print()
print("=" * 80)
print("ANÁLISE DE HIPERPARÂMETROS - KTH (APENAS DESENVOLVIMENTO)")
print("=" * 80)
print(f"Pasta de saída: {OUTPUT_DIR}")
print()



# ============================================================
# 4. CLASSES (KTH)
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
# 5. SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 6. CARREGAR FEATURES (KTH .npy)
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

print()
print("Informações do dataset:")
print("X:", X.shape)
print("y:", y.shape)
print("Pessoas:", len(np.unique(persons)))
print("Sequências:", len(X))
print("Classes mapeadas:", list(CLASS_MAP.keys()))


# ============================================================
# 6. SALVAR E IMPRIMIR PESSOAS DOS CONJUNTOS PRINCIPAIS (DEV / TESTE)
# ============================================================

splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
development_idx, final_test_idx = next(splitter.split(X, y, groups=persons))

persons_dev_unique = sorted(list(set(persons[development_idx])))
persons_test_unique = sorted(list(set(persons[final_test_idx])))

# Verificações de integridade do split
if set(persons_dev_unique) & set(persons_test_unique):
    raise RuntimeError("Erro: Pessoa presente simultaneamente em desenvolvimento e teste!")

if set(np.unique(y[development_idx])) != set(range(N_CLASSES)):
    raise RuntimeError("Erro: O conjunto de desenvolvimento não contém todas as classes!")

# Salvando IMEDIATAMENTE as pessoas de desenvolvimento e teste separadamente
pd.DataFrame({"pessoa": persons_dev_unique}).to_csv(
    OUTPUT_DIR / "pessoas_desenvolvimento.csv", index=False
)
pd.DataFrame({"pessoa": persons_test_unique}).to_csv(
    OUTPUT_DIR / "pessoas_teste.csv", index=False
)

# Salvando índices numéricos para replicar exatamente os mesmos dados
np.savez_compressed(
    OUTPUT_DIR / "divisao_indices.npz",
    development_idx=development_idx,
    final_test_idx=final_test_idx,
)

pd.DataFrame({
    "conjunto": ["desenvolvimento", "teste_reservado"],
    "n_amostras": [len(development_idx), len(final_test_idx)],
    "n_pessoas": [len(persons_dev_unique), len(persons_test_unique)],
    "pessoas": [",".join(persons_dev_unique), ",".join(persons_test_unique)],
    "seed": [SEED, SEED],
}).to_csv(OUTPUT_DIR / "divisao_pessoas_resumo.csv", index=False)

print()
print("=" * 80)
print("ETAPA 1: DIVISÃO PRINCIPAL (DESENVOLVIMENTO E TESTE)")
print("=" * 80)
print(f"Pessoas em DESENVOLVIMENTO ({len(persons_dev_unique)}):")
print(f"  {', '.join(persons_dev_unique)}")
print(f"\nPessoas em TESTE RESERVADO ({len(persons_test_unique)}):")
print(f"  {', '.join(persons_test_unique)}")
print("\n[INFO] Arquivos 'pessoas_desenvolvimento.csv' e 'pessoas_teste.csv' gerados com sucesso.")

# Filtro estrito: a partir daqui usaremos SOMENTE o conjunto de desenvolvimento
X_dev = X[development_idx]
y_dev = y[development_idx]
persons_dev = persons[development_idx]


# ============================================================
# 7. SALVAR E IMPRIMIR PESSOAS DOS FOLDS (CV EM DESENVOLVIMENTO)
# ============================================================

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(cv.split(X_dev, y_dev, groups=persons_dev))

fold_people_rows = []

print()
print("=" * 80)
print("ETAPA 2: DIVISÃO DOS FOLDS DA VALIDAÇÃO CRUZADA (SOMENTE DESENVOLVIMENTO)")
print("=" * 80)

for fold, (train_idx, val_idx) in enumerate(folds, start=1):
    train_people = sorted(list(set(persons_dev[train_idx])))
    val_people = sorted(list(set(persons_dev[val_idx])))
    
    if set(train_people) & set(val_people):
        raise RuntimeError(f"Sobreposição de pessoas no fold {fold}")
        
    fold_people_rows.append({
        "fold": fold,
        "train_people": ",".join(train_people),
        "validation_people": ",".join(val_people),
        "n_train_people": len(train_people),
        "n_validation_people": len(val_people),
        "n_train_samples": len(train_idx),
        "n_validation_samples": len(val_idx),
    })

    print(f"\n[FOLD {fold}/{N_SPLITS}]")
    print(f"  Pessoas Treino ({len(train_people)}): {', '.join(train_people)}")
    print(f"  Pessoas Validação ({len(val_people)}): {', '.join(val_people)}")

pd.DataFrame(fold_people_rows).to_csv(OUTPUT_DIR / "pessoas_folds.csv", index=False)
print("\n[INFO] Arquivo 'pessoas_folds.csv' gerado com sucesso.")


# ============================================================
# 8. GERAR CONFIGURAÇÕES DE HIPERPARÂMETROS
# ============================================================

def generate_configs(grid):
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    configs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)
    return configs


all_configs = {
    "logistic": generate_configs(LOGISTIC_GRID),
    "svm": generate_configs(SVM_GRID),
    "adaboost": generate_configs(ADABOOST_GRID),
    "xgboost": generate_configs(XGBOOST_GRID),
}

config_rows = []
for model_name, configurations in all_configs.items():
    for params in configurations:
        config_rows.append({"config_id": len(config_rows) + 1,
                            "modelo": model_name, **params})

configs_df = pd.DataFrame(config_rows)
configs_df.to_csv(OUTPUT_DIR / "configuracoes_hiperparametros.csv", index=False)


# ============================================================
# 9. CRIAR MODELO E MÉTRICAS
# ============================================================

def create_model(model_name, params):

    if model_name == "logistic":
        return LogisticRegression(**params, random_state=SEED)

    if model_name == "svm":
        return SVC(**params, probability=True, random_state=SEED)

    if model_name == "adaboost":
        estimator = DecisionTreeClassifier(
            max_depth=params["max_depth"],
            random_state=SEED
        )
        return AdaBoostClassifier(
            estimator=estimator,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=SEED
        )

    if model_name == "xgboost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=N_CLASSES,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=SEED,
            eval_metric="mlogloss",
            n_jobs=1,
            tree_method="hist",
        )

    raise ValueError(f"Modelo desconhecido: {model_name}")


def calculate_metrics(y_true, y_pred, probabilities):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=np.arange(N_CLASSES)
        )
    except ValueError:
        metrics["auc"] = np.nan

    return metrics


# ============================================================
# 10. INÍCIO DOS EXPERIMENTOS DE TREINAMENTO
# ============================================================

print()
print("=" * 80)
print("INICIANDO EXPERIMENTOS E AVALIAÇÕES NOS FOLDS")
print("=" * 80)

results = []
confusion_matrices = []
cv_predictions = []

experiment_start = time.perf_counter()


def run_configuration(config_id, model_name, params):

    print()
    print("-" * 80)
    print(f"Configuração {config_id} | {model_name.upper()}")
    print(params)
    print("-" * 80)

    for fold, (train_idx, val_idx) in enumerate(folds, start=1):

        fold_start = time.perf_counter()

        X_train, y_train = X_dev[train_idx], y_dev[train_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = create_model(model_name, params)

        train_start = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        train_time = time.perf_counter() - train_start

        prediction_start = time.perf_counter()
        y_pred = model.predict(X_val_scaled)
        probabilities = model.predict_proba(X_val_scaled)
        prediction_time = time.perf_counter() - prediction_start

        metrics = calculate_metrics(y_val, y_pred, probabilities)
        fold_time = time.perf_counter() - fold_start

        row = {
            "config_id": config_id,
            "modelo": model_name,
            "fold": fold,
            **params,
            **metrics,
            "train_time_seconds": train_time,
            "prediction_time_seconds": prediction_time,
            "fold_time_seconds": fold_time,
            "n_train_samples": len(train_idx),
            "n_val_samples": len(val_idx),
            "n_train_people": len(np.unique(persons_dev[train_idx])),
            "n_val_people": len(np.unique(persons_dev[val_idx])),
        }

        results.append(row)

        for sample_idx, truth, guess, proba in zip(
            val_idx, y_val, y_pred, probabilities
        ):
            cv_predictions.append({
                "config_id": config_id,
                "modelo": model_name,
                "fold": fold,
                "sample_index": int(sample_idx),
                "y_true": int(truth),
                "y_pred": int(guess),
                "probabilities": np.asarray(proba, dtype=float),
            })

        cm = confusion_matrix(y_val, y_pred, labels=np.arange(N_CLASSES))
        cm_rows = []
        class_names = list(CLASS_MAP.keys())

        for true_idx in range(N_CLASSES):
            for pred_idx in range(N_CLASSES):
                cm_rows.append({
                    "config_id": config_id,
                    "modelo": model_name,
                    "fold": fold,
                    "classe_real": class_names[true_idx],
                    "classe_predita": class_names[pred_idx],
                    "quantidade": int(cm[true_idx, pred_idx]),
                })

        confusion_matrices.extend(cm_rows)

        print(
            f"  Fold {fold}/{N_SPLITS} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"Treino={train_time:.2f}s"
        )


for row in config_rows:
    params = next(
        p for p in all_configs[row["modelo"]]
        if all(row[k] == v for k, v in p.items())
    )
    run_configuration(row["config_id"], row["modelo"], params)


# ============================================================
# 11. CONSOLIDAR RESULTADOS
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "resultados_por_fold.csv", index=False)

summary = (
    results_df
    .groupby(["config_id", "modelo"])
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        train_time_mean=("train_time_seconds", "mean"),
        train_time_std=("train_time_seconds", "std"),
        total_train_time=("train_time_seconds", "sum"),
        prediction_time_mean=("prediction_time_seconds", "mean"),
    )
    .reset_index()
)

summary = summary.merge(
    configs_df,
    on=["config_id", "modelo"],
    how="left",
    suffixes=("", "_config")
)

pooled_rows = []
for config_id, model_name in ((r["config_id"], r["modelo"]) for r in config_rows):
    predictions = [r for r in cv_predictions if r["config_id"] == config_id]
    if len(predictions) != len(X_dev) or len({r["sample_index"] for r in predictions}) != len(X_dev):
        raise RuntimeError(f"Predições ausentes ou duplicadas na configuração {config_id}")
    true = np.array([r["y_true"] for r in predictions])
    guessed = np.array([r["y_pred"] for r in predictions])
    probabilities = np.stack([r["probabilities"] for r in predictions])
    pooled_rows.append({
        "config_id": config_id,
        "modelo": model_name,
        "correct_total": int(np.sum(true == guessed)),
        "samples_total": len(true),
        **{f"{key}_pooled": value for key, value in calculate_metrics(true, guessed, probabilities).items()},
    })

pooled_df = pd.DataFrame(pooled_rows)
pooled_df.to_csv(OUTPUT_DIR / "metricas_agregadas_cv.csv", index=False)

summary = summary.merge(pooled_df.drop(columns="modelo"), on="config_id", how="left")
summary = summary.sort_values(["modelo", "f1_pooled"], ascending=[True, False])
summary.to_csv(OUTPUT_DIR / "resumo_configuracoes.csv", index=False)

best_rows = []
for model_name in all_configs:
    model_summary = summary[summary["modelo"] == model_name]
    if model_summary.empty:
        continue
    best_rows.append(model_summary.iloc[0].to_dict())

best_df = pd.DataFrame(best_rows)
best_df.to_csv(OUTPUT_DIR / "melhores_configuracoes.csv", index=False)


# ============================================================
# 12. MATRIZES DE CONFUSÃO E FINALIZAÇÃO
# ============================================================

def save_confusion_graphs(matrix, title, stem):
    classes = list(CLASS_MAP)
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) != 0
    )
    
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        OUTPUT_DIR / f"{stem}_bruta.csv", index_label="classe_real"
    )
    pd.DataFrame(normalized, index=classes, columns=classes).to_csv(
        OUTPUT_DIR / f"{stem}_normalizada.csv", index_label="classe_real"
    )
    
    for values, suffix, cmap, fmt in (
        (matrix, "bruta", "Blues", "d"),
        (normalized, "normalizada", "Blues", ".2f"),
    ):
        fig, ax = plt.subplots(figsize=(12, 10))
        plot = ax.imshow(values, cmap=cmap, vmin=0, vmax=1 if suffix == "normalizada" else None)
        fig.colorbar(plot, ax=ax)
        ax.set_xticks(range(N_CLASSES), labels=classes, rotation=45, ha="right")
        ax.set_yticks(range(N_CLASSES), labels=classes)
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe real")
        ax.set_title(f"{title} — matriz {suffix}")
        
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                value = values[i, j]
                ax.text(
                    j, i, format(value, fmt), ha="center", va="center",
                    fontsize=7, color="white" if value > values.max() / 2 else "black"
                )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{stem}_{suffix}.png", dpi=150)
        plt.close(fig)


for row in config_rows:
    records = [r for r in cv_predictions if r["config_id"] == row["config_id"]]
    matrix = confusion_matrix(
        [r["y_true"] for r in records],
        [r["y_pred"] for r in records],
        labels=np.arange(N_CLASSES)
    )
    save_confusion_graphs(
        matrix,
        f"{row['modelo']} | configuração {row['config_id']} | CV 5 folds (Dev)",
        f"cv_{row['modelo']}_{row['config_id']:03d}"
    )

confusion_df = pd.DataFrame(confusion_matrices)
confusion_df.to_csv(OUTPUT_DIR / "matrizes_confusao.csv", index=False)

total_time = time.perf_counter() - experiment_start

timing_df = pd.DataFrame({
    "timestamp": [timestamp],
    "total_time_seconds": [total_time],
    "n_configs_total": [len(config_rows)],
    "n_splits": [N_SPLITS],
    "seed": [SEED],
})
timing_df.to_csv(OUTPUT_DIR / "tempo_execucao.csv", index=False)

print()
print("=" * 80)
print("ANÁLISE DE HIPERPARÂMETROS FINALIZADA (APENAS DESENVOLVIMENTO)")
print("=" * 80)
print(f"Tempo total: {total_time:.2f} segundos")
print(f"Resultados salvos em: {OUTPUT_DIR}")
print()
print("Melhores configurações na Validação Cruzada (Desenvolvimento):")

for _, row in best_df.iterrows():
    print()
    print(f"{row['modelo'].upper()}")
    print(f"  Configuração: {int(row['config_id'])}")
    print(f"  Accuracy agregada CV: {row['correct_total']}/{row['samples_total']} = {row['accuracy_pooled']:.4f}")
    print(f"  F1 macro agregado CV: {row['f1_pooled']:.4f}")

print()
print("=" * 80)
