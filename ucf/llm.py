from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import re
import time
import cv2
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import dotenv
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize

# ============================================================
# 1. CONFIGURAÇÕES, LOGS E CLIENTE API 
# ============================================================
dotenv.load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
LOG_FILE = "llm.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

if API_KEY == "SUA_CHAVE_API_FREE_AQUI" or not API_KEY:
  raise ValueError(
      "Insira sua API Key do Gemini (obtenha uma gratuita em:"
      " https://ai.studio)"
  )

client = genai.Client(api_key=API_KEY)

SEED = 42
N_SPLITS = 5
MAX_WORKERS = 1  # Necessário para manter o limite de 15 RPM da Cota Gratuita
VIDEO_FOLDER = "./UCF11"  # Pasta com as 11 subpastas do UCF11
CLIPS_CACHE_DIR = "./clips_ucf11_otimizados"
KEYFRAMES_CACHE_DIR = "./keyframes_ucf11"

# Tabela de Mapeamento de Classes UCF11 (11 Classes)
CLASS_MAP = {
    "basketball": 0,
    "biking": 1,
    "diving": 2,
    "golf_swing": 3,
    "horse_riding": 4,
    "soccer_juggling": 5,
    "swing": 6,
    "tennis_swing": 7,
    "trampoline_jumping": 8,
    "volleyball_spiking": 9,
    "walking": 10,
}
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
CLASSES = list(CLASS_MAP.keys())

# Tabela de estimativa de custo
COST_PER_1M_INPUT_TOKENS = 0.075
COST_PER_1M_OUTPUT_TOKENS = 0.30
ESTIMATED_TOKENS_PER_VIDEO_SEC = 258


# ============================================================
# 2. SCHEMA PYDANTIC PARA UCF11
# ============================================================
class ProbabilidadesUCF11(BaseModel):
  basketball: float = Field(ge=0.0, le=1.0)
  biking: float = Field(ge=0.0, le=1.0)
  diving: float = Field(ge=0.0, le=1.0)
  golf_swing: float = Field(ge=0.0, le=1.0)
  horse_riding: float = Field(ge=0.0, le=1.0)
  soccer_juggling: float = Field(ge=0.0, le=1.0)
  swing: float = Field(ge=0.0, le=1.0)
  tennis_swing: float = Field(ge=0.0, le=1.0)
  trampoline_jumping: float = Field(ge=0.0, le=1.0)
  volleyball_spiking: float = Field(ge=0.0, le=1.0)
  walking: float = Field(ge=0.0, le=1.0)


class RespostaClassificacaoUCF11(BaseModel):
  categoria_predita: str
  probabilidades: ProbabilidadesUCF11


# ============================================================
# 3. OTIMIZAÇÃO DE VÍDEOS E EXTRAÇÃO DE KEYFRAMES
# ============================================================
def converter_para_mp4_leve(
    caminho_entrada, caminho_saida, n_frames_target=10, fps_saida=5.0
):
  """Extrai apenas 10 frames espaçados do vídeo. Reduz o tamanho e o consumo de tokens em ~85%."""
  cap = cv2.VideoCapture(caminho_entrada)
  if not cap.isOpened():
    return False

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  if total_frames <= 0 or width == 0 or height == 0:
    cap.release()
    return False

  indices = np.linspace(0, total_frames - 1, num=n_frames_target, dtype=int)
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(caminho_saida, fourcc, fps_saida, (width, height))

  for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
      out.write(frame)

  cap.release()
  out.release()
  return True


def extrair_keyframe_jpg(caminho_video, caminho_saida_jpg):
  """Extrai o frame central de um vídeo para servir como referência ultra-barata (imagem JPEG)."""
  cap = cv2.VideoCapture(caminho_video)
  if not cap.isOpened():
    return False

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frame_mid = max(0, total_frames // 2)

  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_mid)
  ret, frame = cap.read()
  if ret:
    cv2.imwrite(caminho_saida_jpg, frame)

  cap.release()
  return ret


def preparar_dataset_ucf11():
  os.makedirs(CLIPS_CACHE_DIR, exist_ok=True)
  os.makedirs(KEYFRAMES_CACHE_DIR, exist_ok=True)
  clip_paths, y, groups = [], [], []

  logging.info(
      f"Escaneando e otimizando dataset UCF11 a partir de '{VIDEO_FOLDER}'..."
  )

  if not os.path.exists(VIDEO_FOLDER):
    raise FileNotFoundError(
        f"A pasta '{VIDEO_FOLDER}' não foi encontrada! Verifique o caminho."
    )

  contador = 0
  for root, dirs, files in os.walk(VIDEO_FOLDER):
    for file in files:
      if not file.lower().endswith(
          (".avi", ".mpg", ".mpeg", ".mp4", ".mkv", ".wmv")
      ):
        continue

      caminho_orig = os.path.join(root, file)

      classe_encontrada = None
      caminho_partes = caminho_orig.lower().replace("\\", "/").split("/")
      for c in CLASSES:
        if c in caminho_partes or any(c in p for p in caminho_partes):
          classe_encontrada = c
          break

      if not classe_encontrada:
        continue

      class_id = CLASS_MAP[classe_encontrada]

      match_group = re.search(r"v_[a-zA-Z0-9]+_(\d+)", file)
      if match_group:
        group_id = f"{classe_encontrada}_g{match_group.group(1)}"
      else:
        group_id = f"{classe_encontrada}_{os.path.basename(root)}"

      clip_name_anon = f"ucf11_cls{class_id}_{os.path.splitext(file)[0]}.mp4"
      clip_path_anon = os.path.join(CLIPS_CACHE_DIR, clip_name_anon)

      if not os.path.exists(clip_path_anon):
        sucesso = converter_para_mp4_leve(caminho_orig, clip_path_anon)
        if not sucesso:
          continue

      if os.path.exists(clip_path_anon):
        clip_paths.append(clip_path_anon)
        y.append(class_id)
        groups.append(group_id)
        contador += 1

      if contador % 100 == 0:
        logging.info(f"  [Preparo Dataset] {contador} vídeos compactados...")

  logging.info(
      f"Dataset pronto! {len(clip_paths)} vídeos compactados em 10 frames cada."
  )
  return np.array(clip_paths), np.array(y, dtype=np.int64), np.array(groups)


# ============================================================
# 4. FEW-SHOT OTIMIZADO VIA IMAGENS (KEYFRAMES)
# ============================================================
def carregar_referencias_few_shot_imagens(clip_paths_train, y_train):
  """Usa IMAGENS (JPEGs) em vez de VÍDEOS como referência Few-Shot para economizar 90% dos tokens."""
  referencias = []
  logging.info(
      "  [Few-Shot Otimizado] Carregando 11 Imagens de Referência (1 por"
      " classe)..."
  )

  for action_name, action_id in CLASS_MAP.items():
    idxs = np.where(y_train == action_id)[0]
    if len(idxs) > 0:
      caminho_vid_ref = clip_paths_train[idxs[0]]
      jpg_path = os.path.join(
          KEYFRAMES_CACHE_DIR, f"ref_img_{action_name}.jpg"
      )

      if not os.path.exists(jpg_path):
        extrair_keyframe_jpg(caminho_vid_ref, jpg_path)

      ref_file = client.files.upload(file=jpg_path)
      referencias.append((ref_file, action_name))
      logging.info(f"    -> Frame de Referência OK: {action_name}")

  return referencias


def deletar_referencias_few_shot(referencias):
  for ref_file, _ in referencias:
    try:
      client.files.delete(name=ref_file.name)
    except Exception:
      pass


# ============================================================
# 5. WORKER DE INFERÊNCIA DA API
# ============================================================
def obter_prompt_zero_shot_ucf11():
  classes_str = "\n".join([f"- {c}" for c in CLASSES])
  return f"""
    Você é um especialista em reconhecimento de ações humanas (HAR) em vídeos.
    Analise o vídeo enviado e classifique a ação realizada em EXATAMENTE uma das opções:
{classes_str}

    Retorne a categoria predita e as probabilidades normalizadas de cada classe.
    """


def classificar_video_worker(
    caminho_video: str,
    prompt: str,
    referencias_few_shot: list = None,
    max_retries: int = 10,
) -> dict:
  inicio_execucao = time.time()
  nome_arq = os.path.basename(caminho_video)
  test_file = None

  for tentativa in range(max_retries):
    try:
      test_file = client.files.upload(file=caminho_video)
      break
    except Exception as e:
      err_str = str(e)
      if (
          ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str)
          and tentativa < max_retries - 1
      ):
        espera = 10 * (tentativa + 1)
        logging.warning(
            f"  [Aviso Limite] Aguardando {espera}s para upload de"
            f" {nome_arq}..."
        )
        time.sleep(espera)
      else:
        raise e

  while test_file.state.name == "PROCESSING":
    time.sleep(1)
    test_file = client.files.get(name=test_file.name)

  if test_file.state.name != "ACTIVE":
    try:
      client.files.delete(name=test_file.name)
    except Exception:
      pass
    raise RuntimeError(f"Erro no vídeo: {test_file.state.name}")

  contents = []
  if referencias_few_shot:
    for ref_file, action_name in referencias_few_shot:
      contents.append(ref_file)
      contents.append(f"Imagem de exemplo da ação '{action_name}'.")

  contents.append(test_file)
  contents.append(prompt)

  try:
    for tentativa in range(max_retries):
      try:
        response = client.models.generate_content(
          model="gemini-3.6-flash",
          contents=contents,  # <- Certifique-se de manter 'contents=' explicitamente
          config=types.GenerateContentConfig(
              response_mime_type="application/json",
              response_schema=RespostaClassificacaoUCF11,
              temperature=0.1,
          ),
      )
        texto_bruto_retornado = response.text
        dados = json.loads(texto_bruto_retornado)

        probs_dict = dados.get("probabilidades", {})
        probs_array = [probs_dict.get(c, 0.0) for c in CLASSES]

        soma = sum(probs_array)
        if soma > 0:
          probs_array = [p / soma for p in probs_array]

        tempo_inferencia_s = time.time() - inicio_execucao

        # Pausa de 4.5 segundos garante que ficamos dentro da cota gratuita de 15 RPM
        time.sleep(4.5)

        return {
            "predicao_str": dados.get("categoria_predita", ""),
            "predicao_id": CLASS_MAP.get(dados.get("categoria_predita", ""), -1),
            "probs_dict": probs_dict,
            "probs_array": probs_array,
            "raw_response": texto_bruto_retornado,
            "tempo_s": round(tempo_inferencia_s, 2),
        }

      except Exception as e:
        err_str = str(e)
        if (
            ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str)
            and tentativa < max_retries - 1
        ):
          espera = 15 * (tentativa + 1)
          logging.warning(
              f"  [Aviso 429/503] Cota excedida em {nome_arq}. Aguardando"
              f" {espera}s..."
          )
          time.sleep(espera)
        else:
          raise e
  finally:
    if test_file:
      try:
        client.files.delete(name=test_file.name)
      except Exception:
        pass


# ============================================================
# 6. MÉTRICAS AVANÇADAS
# ============================================================
def calcular_metricas_completas(
    y_true, y_pred, y_probs, tempo_total_s, n_amostras, modo_nome
):
  acc = accuracy_score(y_true, y_pred)
  prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
      y_true, y_pred, average="macro", zero_division=0
  )
  prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
      y_true, y_pred, average="weighted", zero_division=0
  )

  try:
    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
    y_probs_arr = np.array(y_probs)
    auc_roc_macro = roc_auc_score(
        y_true_bin, y_probs_arr, multi_class="ovr", average="macro"
    )
    auc_roc_weighted = roc_auc_score(
        y_true_bin, y_probs_arr, multi_class="ovr", average="weighted"
    )
  except Exception as e:
    auc_roc_macro, auc_roc_weighted = None, None

  cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
  tempo_por_100 = (tempo_total_s / n_amostras) * 100 if n_amostras > 0 else 0

  tokens_entrada_est = n_amostras * (
      2.0 * ESTIMATED_TOKENS_PER_VIDEO_SEC + 150
  )  # Vídeos curtos de ~2s
  tokens_saida_est = n_amostras * 100

  custo_est_usd = (
      tokens_entrada_est / 1_000_000
  ) * COST_PER_1M_INPUT_TOKENS + (
      tokens_saida_est / 1_000_000
  ) * COST_PER_1M_OUTPUT_TOKENS

  return {
      "modo": modo_nome,
      "total_amostras": int(n_amostras),
      "acuracia": round(float(acc), 4),
      "precisao_macro": round(float(prec_macro), 4),
      "precisao_weighted": round(float(prec_weight), 4),
      "revocacao_recall_macro": round(float(rec_macro), 4),
      "revocacao_recall_weighted": round(float(rec_weight), 4),
      "f1_score_macro": round(float(f1_macro), 4),
      "f1_score_weighted": round(float(f1_weight), 4),
      "auc_roc_macro": (
          round(float(auc_roc_macro), 4)
          if auc_roc_macro is not None
          else "N/A"
      ),
      "auc_roc_weighted": (
          round(float(auc_roc_weighted), 4)
          if auc_roc_weighted is not None
          else "N/A"
      ),
      "tempo_total_segundos": round(float(tempo_total_s), 2),
      "tempo_por_100_amostras_segundos": round(float(tempo_por_100), 2),
      "custo_estimado_usd": round(float(custo_est_usd), 6),
  }, cm


# ============================================================
# 7. EXPERIMENTO K-FOLD COM CHECKPOINTS INCREMENTAIS
# ============================================================
def rodar_experimento_kfold(
    clip_paths, y, groups, modo="zero-shot", max_workers=MAX_WORKERS
):
  logging.info("\n" + "=" * 70)
  logging.info(
      f" AVALIAÇÃO K-FOLD UCF11 ({N_SPLITS} FOLDS) | MODO: {modo.upper()}"
  )
  logging.info("=" * 70)

  cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
  y_true_global, y_pred_global, y_probs_global = [], [], []
  tempo_inicio_global = time.time()

  for fold, (train_idx, val_idx) in enumerate(
      cv.split(clip_paths, y, groups=groups), start=1
  ):
    logging.info(f"\n>>> INICIANDO FOLD {fold}/{N_SPLITS}...")

    csv_fold_pred_path = f"predicoes_ucf11_{modo}_fold_{fold}.csv"
    json_fold_metricas_path = f"metricas_ucf11_{modo}_fold_{fold}.json"
    csv_fold_cm_path = f"matriz_confusao_ucf11_{modo}_fold_{fold}.csv"

    if os.path.exists(csv_fold_pred_path):
      df_fold_hist = pd.read_csv(csv_fold_pred_path)
      logging.info(
          f"  [Checkpoint] Reaproveitando {len(df_fold_hist)} amostras salvas"
          f" do Fold {fold}."
      )
    else:
      df_fold_hist = pd.DataFrame()

    clip_paths_train, y_train = clip_paths[train_idx], y[train_idx]
    clip_paths_val, y_val, groups_val = (
        clip_paths[val_idx],
        y[val_idx],
        groups[val_idx],
    )

    referencias_few_shot = None
    if modo == "few-shot":
      referencias_few_shot = carregar_referencias_few_shot_imagens(
          clip_paths_train, y_train
      )
      prompt = f"""
            Analise o vídeo enviado e classifique a ação humana realizada
            com base estrita nas imagens de referência fornecidas para cada classe.

            As classes possíveis são:
            {', '.join(CLASSES)}

            Retorne a categoria predita e as probabilidades normalizadas de cada classe.
            """
    else:
      prompt = obter_prompt_zero_shot_ucf11()

    y_true_fold, y_pred_fold, y_probs_fold = [], [], []
    tempo_inicio_fold = time.time()

    try:

      def processar_item(caminho_vid, y_r, g_id):
        if (
            not df_fold_hist.empty
            and "caminho_video" in df_fold_hist.columns
            and caminho_vid in df_fold_hist["caminho_video"].values
        ):
          row = df_fold_hist[
              df_fold_hist["caminho_video"] == caminho_vid
          ].iloc[0]
          probs_arr = [
              float(row[f"prob_{c}"])
              if f"prob_{c}" in row
              else 0.0
              for c in CLASSES
          ]
          return {
              "caminho": caminho_vid,
              "y_real": y_r,
              "res": {
                  "predicao_id": int(row["predicao_id"]),
                  "predicao_str": str(row["predicao_str"]),
                  "probs_array": probs_arr,
                  "probs_dict": {c: float(row.get(f"prob_{c}", 0.0)) for c in CLASSES},
                  "raw_response": str(row.get("raw_response", "")),
                  "tempo_s": float(row["tempo_s"]),
              },
              "cache": True,
          }

        logging.info(f"  [->] Processando: {os.path.basename(caminho_vid)}...")
        res = classificar_video_worker(
            caminho_vid, prompt, referencias_few_shot=referencias_few_shot
        )
        return {
            "caminho": caminho_vid,
            "y_real": y_r,
            "res": res,
            "cache": False,
        }

      with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(processar_item, c_vid, y_r, g_id)
            for c_vid, y_r, g_id in zip(clip_paths_val, y_val, groups_val)
        ]

        concluidos = 0
        total_val = len(futures)

        for future in as_completed(futures):
          concluidos += 1
          try:
            item = future.result()
            res = item["res"]
            y_real = item["y_real"]
            caminho_vid = item["caminho"]

            y_true_fold.append(y_real)
            y_pred_fold.append(res["predicao_id"])
            y_probs_fold.append(res["probs_array"])

            tag_cache = " [CACHE]" if item["cache"] else ""
            logging.info(
                f"  [OK {concluidos}/{total_val}]{tag_cache}"
                f" {os.path.basename(caminho_vid)} | Real:"
                f" {ID_TO_CLASS[y_real]} -> Pred: {res['predicao_str']} | Tempo:"
                f" {res['tempo_s']}s"
            )

            if not item["cache"]:
              linha_dict = {
                  "caminho_video": caminho_vid,
                  "y_real": y_real,
                  "classe_real_str": ID_TO_CLASS[y_real],
                  "predicao_id": res["predicao_id"],
                  "predicao_str": res["predicao_str"],
                  "raw_response": res["raw_response"],
                  "tempo_s": res["tempo_s"],
              }

              for c in CLASSES:
                linha_dict[f"prob_{c}"] = res["probs_dict"].get(c, 0.0)

              linha_dict["probs_json"] = json.dumps(res["probs_array"])

              nova_linha = pd.DataFrame([linha_dict])
              header_necessario = not os.path.exists(csv_fold_pred_path)
              nova_linha.to_csv(
                  csv_fold_pred_path,
                  mode="a",
                  header=header_necessario,
                  index=False,
              )

          except Exception as e:
            logging.error(f"  [ERRO] Falha ao processar amostra: {e}")

      tempo_total_fold_s = time.time() - tempo_inicio_fold
      metricas_fold, cm_fold = calcular_metricas_completas(
          y_true_fold,
          y_pred_fold,
          y_probs_fold,
          tempo_total_fold_s,
          len(y_true_fold),
          f"{modo}_fold_{fold}",
      )

      with open(json_fold_metricas_path, "w", encoding="utf-8") as f:
        json.dump(metricas_fold, f, indent=4, ensure_ascii=False)

      df_cm_fold = pd.DataFrame(cm_fold, index=CLASSES, columns=CLASSES)
      df_cm_fold.to_csv(csv_fold_cm_path)

      logging.info(
          f"  [Fold {fold} Concluído] Acurácia: {metricas_fold['acuracia']} |"
          f" F1-Weighted: {metricas_fold['f1_score_weighted']}"
      )

      y_true_global.extend(y_true_fold)
      y_pred_global.extend(y_pred_fold)
      y_probs_global.extend(y_probs_fold)

    finally:
      if referencias_few_shot:
        deletar_referencias_few_shot(referencias_few_shot)

  tempo_total_experimento_s = time.time() - tempo_inicio_global
  metricas_globais, cm_global = calcular_metricas_completas(
      y_true_global,
      y_pred_global,
      y_probs_global,
      tempo_total_experimento_s,
      len(y_true_global),
      f"{modo}_global_consolidado",
  )

  json_global_path = f"resumo_metricas_globais_ucf11_{modo}.json"
  csv_global_cm_path = f"matriz_confusao_ucf11_{modo}_global.csv"

  with open(json_global_path, "w", encoding="utf-8") as f:
    json.dump(metricas_globais, f, indent=4, ensure_ascii=False)

  df_cm_global = pd.DataFrame(cm_global, index=CLASSES, columns=CLASSES)
  df_cm_global.to_csv(csv_global_cm_path)

  logging.info("\n" + "=" * 70)
  logging.info(f" RESULTADOS GLOBAIS CONSOLIDADOS UCF11 | MODO: {modo.upper()}")
  logging.info("=" * 70)
  for k, v in metricas_globais.items():
    logging.info(f" - {k}: {v}")


# ============================================================
# 8. EXECUÇÃO
# ============================================================
if __name__ == "__main__":
  clip_paths, y, groups = preparar_dataset_ucf11()

  # Executa o modo Zero-Shot (Gratuito)
  rodar_experimento_kfold(clip_paths, y, groups, modo="zero-shot")

  # Executa o modo Few-Shot Otimizado por Imagens (Gratuito)
  rodar_experimento_kfold(clip_paths, y, groups, modo="few-shot")
