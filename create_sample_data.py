import os
import sys
import csv
import numpy as np  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.utils import LABEL_LIST, extract_landmarks  # type: ignore
except Exception:
    from utils import LABEL_LIST, extract_landmarks  # type: ignore

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "hand_landmarks.csv")

# Real Human MediaPipe Skeleton Generator Base
# 21 Keypoints: 0=Wrist, 1-4=Thumb, 5-8=Index, 9-12=Middle, 13-16=Ring, 17-20=Pinky
BASE_OPEN_HAND = np.array([
    [0.0, 0.0, 0.0],         # 0: Wrist
    [0.10, -0.08, -0.02],    # 1: Thumb CMC
    [0.18, -0.16, -0.04],    # 2: Thumb MCP
    [0.25, -0.24, -0.06],    # 3: Thumb IP
    [0.32, -0.30, -0.08],    # 4: Thumb TIP
    [0.08, -0.32, -0.02],    # 5: Index MCP
    [0.09, -0.45, -0.04],    # 6: Index PIP
    [0.10, -0.55, -0.05],    # 7: Index DIP
    [0.10, -0.65, -0.06],    # 8: Index TIP
    [0.00, -0.34, 0.00],     # 9: Middle MCP
    [0.00, -0.48, -0.02],    # 10: Middle PIP
    [0.00, -0.58, -0.03],    # 11: Middle DIP
    [0.00, -0.68, -0.04],    # 12: Middle TIP
    [-0.08, -0.32, -0.01],   # 13: Ring MCP
    [-0.09, -0.45, -0.03],   # 14: Ring PIP
    [-0.10, -0.54, -0.04],   # 15: Ring DIP
    [-0.10, -0.63, -0.05],   # 16: Ring TIP
    [-0.16, -0.28, -0.02],   # 17: Pinky MCP
    [-0.18, -0.38, -0.04],   # 18: Pinky PIP
    [-0.19, -0.46, -0.05],   # 19: Pinky DIP
    [-0.20, -0.54, -0.06],   # 20: Pinky TIP
], dtype=np.float32)


def generate_sign_skeleton(label):
    """
    Generates a realistic 21-landmark 3D hand skeleton for a target sign.
    
    Args:
        label (str): Target ASL sign label.
        
    Returns:
        np.ndarray: (21, 3) 3D coordinate array.
    """
    coords = BASE_OPEN_HAND.copy()

    # Extension states: [Thumb, Index, Middle, Ring, Pinky] (1.0 = Extended, 0.0 = Curled)
    states = {
        'A': [0.6, 0.0, 0.0, 0.0, 0.0],
        'B': [0.0, 1.0, 1.0, 1.0, 1.0],
        'C': [0.55, 0.55, 0.55, 0.55, 0.55],
        'D': [0.10, 1.00, 0.10, 0.10, 0.10],
        'E': [0.00, 0.00, 0.00, 0.00, 0.00],
        'F': [0.20, 0.20, 1.00, 1.00, 1.00],
        'G': [0.95, 0.95, 0.00, 0.00, 0.00],
        'H': [0.15, 0.95, 0.95, 0.00, 0.00],
        'I': [0.00, 0.00, 0.00, 0.00, 1.00],
        'J': [0.00, 0.00, 0.00, 0.00, 0.75],
        'K': [0.50, 1.00, 0.80, 0.00, 0.00],
        'L': [1.00, 1.00, 0.00, 0.00, 0.00],
        'M': [0.20, 0.10, 0.10, 0.10, 0.00],
        'N': [0.20, 0.10, 0.10, 0.00, 0.00],
        'O': [0.30, 0.30, 0.30, 0.30, 0.30],
        'P': [0.50, 0.80, 0.40, 0.00, 0.00],
        'Q': [0.85, 0.40, 0.00, 0.00, 0.00],
        'R': [0.10, 0.95, 0.85, 0.00, 0.00],
        'S': [0.40, 0.00, 0.00, 0.00, 0.00],
        'T': [0.30, 0.10, 0.00, 0.00, 0.00],
        'U': [0.00, 1.00, 0.98, 0.00, 0.00],
        'V': [0.00, 1.00, 0.92, 0.00, 0.00],
        'W': [0.00, 1.00, 0.95, 0.90, 0.00],
        'X': [0.00, 0.40, 0.00, 0.00, 0.00],
        'Y': [1.00, 0.00, 0.00, 0.00, 1.00],
        'Z': [0.00, 0.90, 0.00, 0.00, 0.00],
        'hello': [1.00, 1.00, 1.00, 1.00, 1.00],
        'thank_you': [0.70, 0.70, 0.70, 0.70, 0.70],
        'yes': [1.00, 0.00, 0.00, 0.00, 0.00],
        'no': [0.20, 0.80, 0.80, 0.00, 0.00],
        'please': [0.90, 0.90, 0.90, 0.90, 0.40],
        'space': [1.00, 0.45, 0.45, 0.45, 0.45],
        'delete': [0.00, 0.85, 0.00, 0.00, 0.00],
    }

    ext = states.get(label, [0.5, 0.5, 0.5, 0.5, 0.5])

    # 1. Modify Thumb
    if ext[0] < 0.5:
        coords[3] = coords[2] + np.array([-0.05, 0.02, 0.05])
        coords[4] = coords[2] + np.array([-0.08, 0.05, 0.08])

    # 2. Modify 4 Fingers (Index, Middle, Ring, Pinky)
    finger_indices = [
        [5, 6, 7, 8],     # Index
        [9, 10, 11, 12],  # Middle
        [13, 14, 15, 16], # Ring
        [17, 18, 19, 20]  # Pinky
    ]

    for idx_f, f_indices in enumerate(finger_indices):
        e = ext[idx_f + 1]
        mcp_pos = coords[f_indices[0]]

        if e < 0.5:  # Curled Finger into Palm
            factor = (0.5 - e) * 2.0
            coords[f_indices[1]] = mcp_pos + np.array([0.0, 0.06 * factor, 0.06 * factor])
            coords[f_indices[2]] = mcp_pos + np.array([0.0, 0.04 * factor, 0.12 * factor])
            coords[f_indices[3]] = mcp_pos + np.array([0.0, 0.02 * factor, 0.16 * factor])

    # Custom Spreading & Angles for Specific Signs
    if label == 'A':
        coords[4] = np.array([0.14, -0.16, -0.04]) # Thumb beside index
    elif label == 'C':
        coords[4] = np.array([0.22, -0.22, -0.10]) # Thumb arched C
        coords[8] = np.array([0.18, -0.45, -0.10]) # Index arched C
        coords[12] = np.array([0.10, -0.48, -0.10])
        coords[16] = np.array([-0.02, -0.45, -0.10])
        coords[20] = np.array([-0.12, -0.38, -0.10])
    elif label == 'Z':
        coords[8] = np.array([0.14, -0.68, 0.05])  # Index pointing forward tracing Z
    elif label == 'J':
        coords[20] = np.array([-0.25, -0.55, 0.08]) # Pinky hooked for J
    elif label == 'delete':
        coords[8] = np.array([0.00, -0.45, -0.20]) # Index bent down for backspace
    elif label == 'space':
        coords[4] = np.array([0.45, -0.10, 0.00])  # Thumb extended horizontal
    elif label == 'S':
        coords[4] = np.array([0.02, -0.22, 0.04])  # Thumb over fist
    elif label == 'T':
        coords[4] = np.array([0.06, -0.25, 0.04])  # Thumb under index
    elif label == 'E':
        coords[4] = np.array([0.00, -0.18, 0.06])  # Thumb under all fingertips
    elif label == 'M':
        coords[4] = np.array([-0.08, -0.25, 0.04]) # Thumb under 3 fingers
    elif label == 'N':
        coords[4] = np.array([-0.04, -0.25, 0.04]) # Thumb under 2 fingers
    elif label == 'L':
        coords[4] = np.array([0.35, -0.15, -0.04])  # Thumb far right
        coords[8] = np.array([0.08, -0.68, -0.06])  # Index straight up
    elif label == 'V':
        coords[8] = np.array([0.16, -0.65, -0.05])  # Index right
        coords[12] = np.array([-0.16, -0.68, -0.04]) # Middle left
    elif label == 'U':
        coords[8] = np.array([0.04, -0.66, -0.05])  # Index close
        coords[12] = np.array([-0.04, -0.68, -0.04]) # Middle close
    elif label == 'R':
        coords[8] = np.array([-0.02, -0.66, -0.05]) # Index crossed
        coords[12] = np.array([0.02, -0.68, -0.04])  # Middle crossed
    elif label == 'Y':
        coords[4] = np.array([0.38, -0.20, -0.04])  # Thumb wide right
        coords[20] = np.array([-0.28, -0.54, -0.06]) # Pinky wide left

    return coords


def extract_features_from_skeleton(coords):
    """
    Extracts identical 63-element invariant feature vector from 3D coords.
    """
    wrist = coords[0]
    coords_rel = coords - wrist
    palm_size = np.linalg.norm(coords[5] - coords[0])
    if palm_size > 1e-6:
        coords_norm = coords_rel / palm_size
    else:
        coords_norm = coords_rel

    norm_xy = coords_norm[:, :2].flatten()

    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    ext_ratios = []
    for tip_i, mcp_i in zip(tips, mcps):
        tip_dist = np.linalg.norm(coords_norm[tip_i])
        mcp_dist = np.linalg.norm(coords_norm[mcp_i])
        ratio = tip_dist / (mcp_dist + 1e-5)
        ext_ratios.append(ratio)

    tip_dists = [
        np.linalg.norm(coords_norm[8] - coords_norm[4]),
        np.linalg.norm(coords_norm[8] - coords_norm[12]),
        np.linalg.norm(coords_norm[12] - coords_norm[16]),
        np.linalg.norm(coords_norm[16] - coords_norm[20]),
        np.linalg.norm(coords_norm[4] - coords_norm[20]),
    ]

    v_thumb = coords_norm[4]
    v_index = coords_norm[8]
    v_middle = coords_norm[12]
    v_ring = coords_norm[16]
    v_pinky = coords_norm[20]

    def get_angle(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-5 or n2 < 1e-5:
            return 0.0
        cos_a = np.dot(v1, v2) / (n1 * n2)
        return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    angles = [
        get_angle(v_thumb, v_index),
        get_angle(v_index, v_middle),
        get_angle(v_middle, v_ring),
        get_angle(v_ring, v_pinky),
    ]

    hand_feat = np.zeros(63, dtype=np.float32)
    combined = np.concatenate([norm_xy, ext_ratios, tip_dists, angles])
    hand_feat[:len(combined)] = combined

    return hand_feat


def generate_sample_dataset(samples_per_class=120):
    """
    Generates realistic anatomical dataset for all target sign classes.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    headers = ["label"] + [f"feat_{i}" for i in range(126)]
    total_rows = 0

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for label in LABEL_LIST:
            skel1 = generate_sign_skeleton(label)
            h1_base = extract_features_from_skeleton(skel1)

            if label in ['thank_you', 'please', 'hello']:
                skel2 = generate_sign_skeleton(label)
                h2_base = extract_features_from_skeleton(skel2)
            else:
                h2_base = np.zeros(63, dtype=np.float32)

            combined_base = np.concatenate([h1_base, h2_base])

            for _ in range(samples_per_class):
                noise = np.random.normal(0.0, 0.02, size=combined_base.shape).astype(np.float32)
                sample = combined_base + noise

                if label not in ['thank_you', 'please', 'hello']:
                    sample[63:] = 0.0

                writer.writerow([label] + sample.tolist())
                total_rows += 1

    print(f"[MediaPipe Skeleton Generator] Created dataset at '{CSV_PATH}' with {total_rows} samples.")


if __name__ == "__main__":
    generate_sample_dataset()
