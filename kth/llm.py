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

if API_KEY == "SUA_CHAVE_API_AQUI" or not API_KEY:
  raise ValueError("Por favor, insira sua chave API na variável API_KEY.")

client = genai.Client(api_key=API_KEY)

SEED = 42
N_SPLITS = 5
MAX_WORKERS = 10
VIDEO_FOLDER = "./kth_videos"
INFO_PATH = "./00sequences.txt"
CLIPS_CACHE_DIR = "./clips_kth"

CLASS_MAP = {
    "walking": 0,
    "jogging": 1,
    "running": 2,
    "boxing": 3,
    "handclapping": 4,
    "handwaving": 5,
}
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
CLASSES = list(CLASS_MAP.keys())

COST_PER_1M_INPUT_TOKENS = 0.075
COST_PER_1M_OUTPUT_TOKENS = 0.30
ESTIMATED_TOKENS_PER_VIDEO_SEC = 258


# ============================================================
# 2. SCHEMA PYDANTIC
# ============================================================
class ProbabilidadesKTH(BaseModel):
  walking: float = Field(ge=0.0, le=1.0)
  jogging: float = Field(ge=0.0, le=1.0)
  running: float = Field(ge=0.0, le=1.0)
  boxing: float = Field(ge=0.0, le=1.0)
  handclapping: float = Field(ge=0.0, le=1.0)
  handwaving: float = Field(ge=0.0, le=1.0)


class RespostaClassificacao(BaseModel):
  categoria_predita: str
  probabilidades: ProbabilidadesKTH


# ============================================================
# 3. LEITURA E EXTRAÇÃO
# ============================================================
def read_info(path_info):
  seqs = []
  with open(path_info, "r") as archive:
    for line in archive:
      line = line.strip()
      if not line:
        continue
      parts = line.split()
      vid_name = parts[0]
      text_frames = " ".join(parts[2:])
      intervals = re.findall(r"(\d+)-(\d+)", text_frames)
      for begin, end in intervals:
        seqs.append(
            {"video": vid_name, "begin": int(begin), "end": int(end)}
        )
  return seqs


def extract_frames(video_path, begin, end):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    logging.error(f"Não foi possível abrir o vídeo: {video_path}")
    return None
  cap.set(cv2.CAP_PROP_POS_FRAMES, begin - 1)
  frames = []
  for frame_number in range(begin, end + 1):
    ret, frame = cap.read()
    if not ret:
      logging.error(f"ERRO ao ler frame {frame_number} do vídeo {video_path}")
      cap.release()
      return None
    frames.append(frame)
  cap.release()
  return frames


# ============================================================
# 4. PROCESSAMENTO E ANONIMIZAÇÃO
# ============================================================
def salvar_frames_como_video(frames, output_path, fps=25.0):
  if not frames:
    return False
  height, width, _ = frames[0].shape
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
  for f in frames:
    out.write(f)
  out.release()
  return True


def preparar_dataset_recortado():
  os.makedirs(CLIPS_CACHE_DIR, exist_ok=True)
  seqs = read_info(INFO_PATH)

  mapa_videos_originais = {}
  for root, dirs, files in os.walk(VIDEO_FOLDER):
    for f in files:
      if f.endswith(".avi") or f.endswith(".mp4"):
        nome_limpo = os.path.splitext(f)[0].replace("_uncomp", "")
        mapa_videos_originais[nome_limpo] = os.path.join(root, f)

  clip_paths, y, persons = [], [], []
  total_seqs = len(seqs)
  logging.info(f"Processando {total_seqs} sequências via read_info()...")

  for i, seq in enumerate(seqs, start=1):
    nome_vid = seq["video"]
    if nome_vid not in mapa_videos_originais:
      continue
    caminho_orig = mapa_videos_originais[nome_vid]

    classe = None
    for c in CLASSES:
      if c in nome_vid:
        classe = c
        break
    if not classe:
      continue

    class_id = CLASS_MAP[classe]
    person = nome_vid.split("_")[0]

    nome_vid_anon = nome_vid.replace(classe, str(class_id))
    clip_name = f"{nome_vid_anon}_seq{seq['begin']}_{seq['end']}.mp4"
    clip_path = os.path.join(CLIPS_CACHE_DIR, clip_name)

    if not os.path.exists(clip_path):
      frames = extract_frames(caminho_orig, seq["begin"], seq["end"])
      if frames is not None:
        salvar_frames_como_video(frames, clip_path)

    if i % 100 == 0 or i == total_seqs:
      logging.info(f"  [Recorte] {i}/{total_seqs} clipes verificados...")

    if os.path.exists(clip_path):
      clip_paths.append(clip_path)
      y.append(class_id)
      persons.append(person)

  logging.info(
      f"Dataset pronto! {len(clip_paths)} sequências anonimizadas carregadas."
  )
  return np.array(clip_paths), np.array(y, dtype=np.int64), np.array(persons)


# ============================================================
# 5. WORKER (RETORNA O DADO BRUTO DO PROMPT NA ÍNTEGRA)
# ============================================================
def obter_prompt_zero_shot():
  return """
    Você é um especialista em reconhecimento de ações humanas (HAR).
    Analise o vídeo enviado e classifique a ação em EXATAMENTE uma das opções:
    - walking
    - jogging
    - running
    - boxing
    - handclapping
    - handwaving

    Retorne a categoria predita e as probabilidades normalizadas de cada classe.
    """


def carregar_referencias_few_shot(clip_paths_train, y_train):
  referencias = []
  logging.info("  [Few-Shot] Fazendo upload das referências de treino...")
  for action_name, action_id in CLASS_MAP.items():
    idxs = np.where(y_train == action_id)[0]
    if len(idxs) > 0:
      caminho_ref = clip_paths_train[idxs[0]]
      ref_file = client.files.upload(file=caminho_ref)
      while ref_file.state.name == "PROCESSING":
        time.sleep(1)
        ref_file = client.files.get(name=ref_file.name)
      referencias.append((ref_file, action_name))
      logging.info(f"    -> Referência carregada: {action_name}")
  return referencias


def deletar_referencias_few_shot(referencias):
  for ref_file, _ in referencias:
    try:
      client.files.delete(name=ref_file.name)
    except Exception:
      pass


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
        espera = 15 * (tentativa + 1)
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
      contents.append(f"Exemplo da ação '{action_name}'.")

  contents.append(test_file)
  contents.append(prompt)

  try:
    for tentativa in range(max_retries):
      try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RespostaClassificacao,
                temperature=0.1,
            ),
        )

        # CAPTURA DA RESPOSTA BRUTA DO PROMPT
        texto_bruto_retornado = response.text

        dados = json.loads(texto_bruto_retornado)
        probs_dict = dados.get("probabilidades", {})
        probs_array = [probs_dict.get(c, 0.0) for c in CLASSES]

        soma = sum(probs_array)
        if soma > 0:
          probs_array = [p / soma for p in probs_array]

        tempo_inferencia_s = time.time() - inicio_execucao
        time.sleep(4.0)

        return {
            "predicao_str": dados.get("categoria_predita", ""),
            "predicao_id": CLASS_MAP.get(dados.get("categoria_predita", ""), -1),
            "probs_dict": probs_dict,
            "probs_array": probs_array,
            "raw_response": texto_bruto_retornado,  # DADO BRUTO INTEGRAL SALVO
            "tempo_s": round(tempo_inferencia_s, 2),
        }

      except Exception as e:
        err_str = str(e)
        if (
            ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str)
            and tentativa < max_retries - 1
        ):
          espera = 20 * (tentativa + 1)
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
# 6. MÉTRICAS COMPLETA
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
    logging.warning(f"Não foi possível calcular AUC-ROC: {e}")
    auc_roc_macro, auc_roc_weighted = None, None

  cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
  tempo_por_100_amostras = (
      (tempo_total_s / n_amostras) * 100 if n_amostras > 0 else 0
  )

  duracao_media_video_s = 4.0
  tokens_entrada_est = n_amostras * (
      duracao_media_video_s * ESTIMATED_TOKENS_PER_VIDEO_SEC + 150
  )
  tokens_saida_est = n_amostras * 80
  custo_total_usd = (
      tokens_entrada_est / 1_000_000
  ) * COST_PER_1M_INPUT_TOKENS + (
      tokens_saida_est / 1_000_000
  ) * COST_PER_1M_OUTPUT_TOKENS

  metricas_dict = {
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
      "tempo_por_100_amostras_segundos": round(
          float(tempo_por_100_amostras), 2
      ),
      "custo_estimado_usd": round(float(custo_total_usd), 6),
  }

  return metricas_dict, cm


# ============================================================
# 7. EXECUÇÃO K-FOLD SALVANDO DADOS DO PROMPT E COLUNAS
# ============================================================
def rodar_experimento_kfold(
    clip_paths, y, persons, modo="zero-shot", max_workers=MAX_WORKERS
):
  logging.info("\n" + "=" * 70)
  logging.info(f" AVALIAÇÃO K-FOLD ({N_SPLITS} FOLDS) | MODO: {modo.upper()}")
  logging.info("=" * 70)

  cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
  y_true_global, y_pred_global, y_probs_global = [], [], []
  tempo_inicio_global = time.time()

  for fold, (train_idx, val_idx) in enumerate(
      cv.split(clip_paths, y, groups=persons), start=1
  ):
    logging.info(f"\n>>> INICIANDO FOLD {fold}/{N_SPLITS}...")

    csv_fold_pred_path = f"predicoes_{modo}_fold_{fold}.csv"
    json_fold_metricas_path = f"metricas_{modo}_fold_{fold}.json"
    csv_fold_cm_path = f"matriz_confusao_{modo}_fold_{fold}.csv"

    if os.path.exists(csv_fold_pred_path):
      df_fold_hist = pd.read_csv(csv_fold_pred_path)
      logging.info(
          f"  [Checkpoint] Encontradas {len(df_fold_hist)} amostras salvas no"
          f" Fold {fold}."
      )
    else:
      df_fold_hist = pd.DataFrame()

    clip_paths_train, y_train = clip_paths[train_idx], y[train_idx]
    clip_paths_val, y_val, persons_val = (
        clip_paths[val_idx],
        y[val_idx],
        persons[val_idx],
    )

    referencias_few_shot = None
    if modo == "few-shot":
      referencias_few_shot = carregar_referencias_few_shot(
          clip_paths_train, y_train
      )
      prompt = """
            Analise o último vídeo enviado e classifique a ação humana realizada
            com base estrita nos vídeos de referência fornecidos acima.
            Atenção às velocidades de deslocamento (walking < jogging < running).

            Retorne a categoria predita e as probabilidades normalizadas de cada classe.
            """
    else:
      prompt = obter_prompt_zero_shot()

    y_true_fold, y_pred_fold, y_probs_fold = [], [], []
    tempo_inicio_fold = time.time()

    try:

      def processar_item(caminho_vid, y_r, p_id):
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
            executor.submit(processar_item, c_vid, y_r, p_id)
            for c_vid, y_r, p_id in zip(clip_paths_val, y_val, persons_val)
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

            # GRAVAÇÃO DETALHADA NO CSV DO FOLD
            if not item["cache"]:
              linha_dict = {
                  "caminho_video": caminho_vid,
                  "y_real": y_real,
                  "classe_real_str": ID_TO_CLASS[y_real],
                  "predicao_id": res["predicao_id"],
                  "predicao_str": res["predicao_str"],
                  "raw_response": res["raw_response"],  # TEXTO BRUTO RETORNADO PELO PROMPT
                  "tempo_s": res["tempo_s"],
              }

              # Adiciona cada classe como uma coluna de probabilidade separada
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
            logging.error(f"  [ERRO] Falha no processamento: {e}")

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

  json_global_path = f"resumo_metricas_globais_{modo}.json"
  csv_global_cm_path = f"matriz_confusao_{modo}_global.csv"

  with open(json_global_path, "w", encoding="utf-8") as f:
    json.dump(metricas_globais, f, indent=4, ensure_ascii=False)

  df_cm_global = pd.DataFrame(cm_global, index=CLASSES, columns=CLASSES)
  df_cm_global.to_csv(csv_global_cm_path)

  logging.info("\n" + "=" * 70)
  logging.info(f" RESULTADOS GLOBAIS CONSOLIDADOS | MODO: {modo.upper()}")
  logging.info("=" * 70)
  for k, v in metricas_globais.items():
    logging.info(f" - {k}: {v}")


if __name__ == "__main__":
  clip_paths, y, persons = preparar_dataset_recortado()

  rodar_experimento_kfold(clip_paths, y, persons, modo="zero-shot")
  rodar_experimento_kfold(clip_paths, y, persons, modo="few-shot")