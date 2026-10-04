import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATASET_DIR = "UCF11"
OUTPUT_FILE = "ucf11_resnet_features.npz"

# Quantidade máxima de frames utilizados por vídeo
NUM_FRAMES = 16

VIDEO_EXTENSIONS = (
    ".avi",
    ".mp4",
    ".mpg",
    ".mpeg",
    ".mov"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# RESNET-50
# ============================================================

print("Carregando ResNet-50...")

weights = ResNet50_Weights.DEFAULT

model = resnet50(
    weights=weights
)

# Remove a camada de classificação.
#
# ResNet original:
#
# imagem -> ... -> 2048 -> FC -> 1000 classes
#
# Agora:
#
# imagem -> ... -> 2048
#
model.fc = nn.Identity()

model = model.to(DEVICE)

model.eval()

# Transformações recomendadas para os pesos ImageNet
transform = weights.transforms()

print("ResNet-50 carregada.")


# ============================================================
# ENCONTRAR TODOS OS VÍDEOS
# ============================================================

videos = []

for root, dirs, files in os.walk(DATASET_DIR):

    for file in files:

        if file.lower().endswith(VIDEO_EXTENSIONS):

            video_path = os.path.join(
                root,
                file
            )

            videos.append(video_path)


videos.sort()

print()
print("=" * 60)
print("DATASET")
print("=" * 60)

print(
    "Número de vídeos encontrados:",
    len(videos)
)


# ============================================================
# OBTER CLASSE E PESSOA
# ============================================================
#
# Estrutura:
#
# UCF11/
#   classe/
#       v_classe_person/
#           v_classe_person_view.mpg
#
# Exemplo:
#
# UCF11/basketball/v_shooting_24/v_shooting_24_01.mpg
#
# classe = basketball
# pessoa = 24
#
# ============================================================

def get_class_and_person(video_path):

    relative = os.path.relpath(
        video_path,
        DATASET_DIR
    )

    parts = relative.split(os.sep)

    # --------------------------------------------------------
    # parts[0] = classe
    # parts[1] = pasta da pessoa
    # parts[2] = vídeo
    # --------------------------------------------------------

    if len(parts) < 3:

        raise ValueError(
            f"Estrutura inesperada: {video_path}"
        )

    class_name = parts[0]

    person_folder = parts[1]

    # Exemplo:
    #
    # v_shooting_24
    #
    # split("_"):
    #
    # ["v", "shooting", "24"]
    #
    person_parts = person_folder.split("_")

    if len(person_parts) < 2:

        raise ValueError(
            f"Pasta de pessoa inválida: {person_folder}"
        )

    # A pessoa é a última parte
    person = person_parts[-1]

    return class_name, person


# ============================================================
# LER FRAMES UNIFORMEMENTE
# ============================================================

def read_sampled_frames(
    video_path,
    num_frames=16
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            f"[ERRO] Não foi possível abrir: "
            f"{video_path}"
        )

        return []

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:

        cap.release()

        print(
            f"[ERRO] Vídeo sem frames: "
            f"{video_path}"
        )

        return []

    # --------------------------------------------------------
    # Selecionar frames uniformemente
    # --------------------------------------------------------
    #
    # Exemplo:
    #
    # vídeo = 160 frames
    # NUM_FRAMES = 16
    #
    # serão escolhidos aproximadamente:
    #
    # 0, 10, 21, 31, ...
    #
    # --------------------------------------------------------

    n = min(
        num_frames,
        total_frames
    )

    indices = np.linspace(
        0,
        total_frames - 1,
        n,
        dtype=int
    )

    frames = []

    for idx in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx)
        )

        ret, frame = cap.read()

        if not ret:
            continue

        # BGR -> RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    cap.release()

    return frames


# ============================================================
# EXTRAIR FEATURE DE UM VÍDEO
# ============================================================

def extract_video_feature(video_path):

    frames = read_sampled_frames(
        video_path,
        NUM_FRAMES
    )

    if len(frames) == 0:
        return None

    # --------------------------------------------------------
    # numpy.ndarray -> PIL Image
    # --------------------------------------------------------

    batch = torch.stack(
        [
            transform(Image.fromarray(frame))
            for frame in frames
        ]
    )

    batch = batch.to(DEVICE)

    # --------------------------------------------------------
    # ResNet
    # --------------------------------------------------------

    with torch.no_grad():

        features = model(batch)

    # [n_frames, 2048]
    #
    # média temporal
    #
    # [2048]

    video_feature = features.mean(dim=0)

    return (
        video_feature
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# EXTRAÇÃO
# ============================================================

X = []
y = []
persons = []
video_paths = []

failed = []


print()
print("=" * 60)
print("EXTRAINDO FEATURES")
print("=" * 60)


for video_path in tqdm(
    videos,
    desc="Vídeos"
):

    # --------------------------------------------------------
    # Classe e pessoa
    # --------------------------------------------------------

    try:

        class_name, person = (
            get_class_and_person(
                video_path
            )
        )

    except Exception as e:

        print(
            f"\n[ERRO] Metadados: "
            f"{video_path}"
        )

        print(e)

        failed.append(video_path)

        continue

    # --------------------------------------------------------
    # Feature
    # --------------------------------------------------------

    feature = extract_video_feature(
        video_path
    )

    if feature is None:

        failed.append(video_path)

        continue

    # --------------------------------------------------------
    # Armazenar
    # --------------------------------------------------------

    X.append(feature)

    y.append(class_name)

    persons.append(person)

    video_paths.append(video_path)


# ============================================================
# NUMPY
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y
)

persons = np.asarray(
    persons
)

video_paths = np.asarray(
    video_paths
)


# ============================================================
# SALVAR
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    y=y,
    persons=persons,
    video_paths=video_paths
)


# ============================================================
# RESUMO
# ============================================================

print()
print("=" * 60)
print("EXTRAÇÃO CONCLUÍDA")
print("=" * 60)

print(
    f"Vídeos processados: {len(X)}"
)

print(
    f"Vídeos com erro:    {len(failed)}"
)

print(
    f"X:                  {X.shape}"
)

print(
    f"y:                  {y.shape}"
)

print(
    f"persons:             {persons.shape}"
)

print(
    f"Classes:             {len(np.unique(y))}"
)

print(
    f"Pessoas:             {len(np.unique(persons))}"
)

print(
    f"Feature por vídeo:   {X.shape[1]} dimensões"
)

print()
print(
    f"Arquivo salvo em: {OUTPUT_FILE}"
)


# ============================================================
# VERIFICAÇÃO
# ============================================================

print()
print("=" * 60)
print("EXEMPLO DOS PRIMEIROS VÍDEOS")
print("=" * 60)

for i in range(min(10, len(X))):

    print(
        f"{i:4d} | "
        f"classe={y[i]:15s} | "
        f"pessoa={persons[i]:5s} | "
        f"{video_paths[i]}"
    )