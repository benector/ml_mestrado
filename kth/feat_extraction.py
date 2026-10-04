import cv2
import re
import logging
import numpy as np
import os


# ============================================================
# CONFIGURAÇÃO DO LOG
# ============================================================

logging.basicConfig(
    filename="extracao_frames.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ============================================================
# LEITURA DO ARQUIVO INFO
# ============================================================

def read_info(path_info):

    seqs = []

    with open(path_info, "r") as archive:

        for line in archive:

            line = line.strip()

            if not line:
                continue

            # Exemplo:
            # person01_boxing_d1 frames 1-95, 96-185, 186-245, 246-360

            parts = line.split()

            vid_name = parts[0]

            text_frames = " ".join(parts[2:])

            intervals = re.findall(
                r"(\d+)-(\d+)",
                text_frames
            )

            for begin, end in intervals:

                seqs.append({
                    "video": vid_name,
                    "begin": int(begin),
                    "end": int(end)
                })

    return seqs


# ============================================================
# EXTRAÇÃO DOS FRAMES
# ============================================================

def extract_frames(video_path, begin, end):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        logging.error(
            f"Não foi possível abrir o vídeo: {video_path}"
        )

        return None

    # O arquivo info começa em 1.
    # O OpenCV começa em 0.

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        begin - 1
    )

    frames = []

    for frame_number in range(begin, end + 1):

        ret, frame = cap.read()

        if not ret:

            logging.error(
                f"ERRO ao ler frame {frame_number} "
                f"do vídeo {video_path} "
                f"(sequência {begin}-{end})"
            )

            cap.release()

            return None

        frames.append(frame)

    cap.release()

    return frames


# ============================================================
# HOF - HISTOGRAM OF OPTICAL FLOW
# ============================================================

def extract_hof(frames, grid_size=4, num_bins=8):

    """
    Extrai Histogram of Optical Flow (HOF).

    Para cada par de frames consecutivos:

        1. calcula Optical Flow de Farneback;
        2. obtém magnitude e direção;
        3. divide a imagem em uma grade 4x4;
        4. calcula um histograma de direções em cada célula;
        5. acumula os histogramas ao longo da sequência.

    Retorna:

        vetor HOF com:

        grid_size × grid_size × num_bins

        = 4 × 4 × 8
        = 128 features
    """

    if len(frames) < 2:

        logging.error(
            "Sequência possui menos de 2 frames."
        )

        return None

    # --------------------------------------------------------
    # Dimensões da imagem
    # --------------------------------------------------------

    height, width = frames[0].shape[:2]

    cell_height = height // grid_size
    cell_width = width // grid_size

    # Histograma final
    hof = np.zeros(
        (
            grid_size,
            grid_size,
            num_bins
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Optical Flow entre frames consecutivos
    # --------------------------------------------------------

    for i in range(len(frames) - 1):

        previous = cv2.cvtColor(
            frames[i],
            cv2.COLOR_BGR2GRAY
        )

        current = cv2.cvtColor(
            frames[i + 1],
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Farneback Optical Flow
        # ----------------------------------------------------

        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,    # pyr_scale
            3,      # levels
            15,     # winsize
            3,      # iterations
            5,      # poly_n
            1.2,    # poly_sigma
            0       # flags
        )

        # ----------------------------------------------------
        # Separar componentes do fluxo
        # ----------------------------------------------------

        dx = flow[..., 0]
        dy = flow[..., 1]

        # Magnitude e direção
        magnitude, angle = cv2.cartToPolar(
            dx,
            dy,
            angleInDegrees=False
        )

        # ----------------------------------------------------
        # Converter direção para 0-360 graus
        # ----------------------------------------------------

        angle_degrees = (
            angle * 180 / np.pi
        )

        # ----------------------------------------------------
        # Dividir imagem em células
        # ----------------------------------------------------

        for row in range(grid_size):

            for col in range(grid_size):

                y_start = row * cell_height
                y_end = (row + 1) * cell_height

                x_start = col * cell_width
                x_end = (col + 1) * cell_width

                cell_magnitude = magnitude[
                    y_start:y_end,
                    x_start:x_end
                ]

                cell_angle = angle_degrees[
                    y_start:y_end,
                    x_start:x_end
                ]

                # ------------------------------------------------
                # Histograma das direções
                # ------------------------------------------------

                hist, _ = np.histogram(
                    cell_angle,
                    bins=num_bins,
                    range=(0, 360),
                    weights=cell_magnitude
                )

                # Acumular ao longo do tempo
                hof[row, col] += hist

    # ========================================================
    # NORMALIZAÇÃO
    # ========================================================

    # Transformar a matriz 4x4x8 em vetor
    hof = hof.flatten()

    # Normalização L2
    norm = np.linalg.norm(hof)

    if norm > 0:

        hof = hof / norm

    return hof.astype(np.float32)


# ============================================================
# PROCESSAMENTO
# ============================================================

info = read_info("info.txt")

# Pasta para salvar SOMENTE os HOFs
os.makedirs(
    "hof_features",
    exist_ok=True
)



for seq in info:

    # --------------------------------------------------------
    # LOCALIZAÇÃO DO VÍDEO
    # --------------------------------------------------------

    video_path = seq["video"].split("_")[1]

    video_file = (
        f"{video_path}/"
        f"{seq['video']}_uncomp.avi"
    )

    # --------------------------------------------------------
    # NOME DO ARQUIVO HOF
    # --------------------------------------------------------

    output_name = (
        f"{seq['video']}_"
        f"{seq['begin']}_"
        f"{seq['end']}.npy"
    )

    output_path = os.path.join(
        "hof_features",
        output_name
    )

    # --------------------------------------------------------
    # VERIFICAR SE JÁ FOI PROCESSADO
    # --------------------------------------------------------

    if os.path.exists(output_path):

        logging.info(
            f"HOF já existe, pulando: {output_path}"
        )

        continue

    logging.info(
        f"Processando {seq['video']} | "
        f"frames {seq['begin']}-{seq['end']}"
    )

    # --------------------------------------------------------
    # EXTRAIR FRAMES
    # --------------------------------------------------------

    extracted_frames = extract_frames(
        video_file,
        seq["begin"],
        seq["end"]
    )

    if extracted_frames is None:

        logging.error(
            f"Falha na sequência: "
            f"{seq['video']} "
            f"{seq['begin']}-{seq['end']}"
        )

        continue

    logging.info(
        f"{len(extracted_frames)} frames extraídos"
    )

    # --------------------------------------------------------
    # HOF
    # --------------------------------------------------------

    hof = extract_hof(
        extracted_frames,
        grid_size=4,
        num_bins=8
    )

    if hof is None:

        logging.error(
            f"Falha na extração HOF: "
            f"{seq['video']} "
            f"{seq['begin']}-{seq['end']}"
        )

        continue

    logging.info(
        f"HOF extraído: shape={hof.shape}"
    )

    # --------------------------------------------------------
    # SALVAR HOF
    # --------------------------------------------------------

    np.save(
        output_path,
        hof
    )

    logging.info(
        f"HOF salvo: {output_path}"
    )

    # Liberar memória
    del extracted_frames
    del hof