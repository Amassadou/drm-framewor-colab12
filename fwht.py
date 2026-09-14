from __future__ import annotations

import hashlib
import io
import os

import bchlib
import numpy as np
from PIL import Image

from core.logging import get_logger


try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


logger = get_logger(__name__)


# ============================================================
# CONFIGURATION BCH
# ============================================================

BCH_T = 8
BCH_M = 8


# ============================================================
# BCH
# ============================================================

def create_bch_codec():
    """
    Crée le codec BCH utilisé pour le watermark.

    BCH_T = nombre maximal d'erreurs de bits corrigeables
    BCH_M = paramètre du code BCH
    """
    return bchlib.BCH(BCH_T, m=BCH_M)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convertit un tableau de bits en bytes.

    Si le nombre de bits n'est pas multiple de 8,
    des zéros sont ajoutés à la fin.
    """
    bits = np.asarray(bits, dtype=np.uint8).flatten()

    if len(bits) == 0:
        return b""

    padding = (-len(bits)) % 8

    if padding:
        bits = np.pad(
            bits,
            (0, padding),
            constant_values=0,
        )

    return np.packbits(bits).tobytes()


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Convertit des bytes en tableau de bits.
    """
    if not data:
        return np.array([], dtype=np.uint8)

    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8)
    ).astype(np.uint8)


def bch_encode_bits(
    bits: np.ndarray,
    bch,
) -> np.ndarray:
    """
    Encode les bits du watermark avec BCH.

    Les données sont découpées en blocs BCH.

    Pour BCH(8,8) :
        n = 255 bits
        ecc = 8 bytes
        données maximales par bloc = 23 bytes
    """

    data = bits_to_bytes(bits)

    if not data:
        return np.array([], dtype=np.uint8)

    block_size = (bch.n // 8) - bch.ecc_bytes

    if block_size <= 0:
        raise ValueError(
            "Taille de bloc BCH invalide."
        )

    encoded = bytearray()

    for start in range(0, len(data), block_size):

        block = data[
            start:start + block_size
        ]

        if not block:
            continue

        ecc = bch.encode(block)

        encoded.extend(block)
        encoded.extend(ecc)

    return bytes_to_bits(
        bytes(encoded)
    )


def bch_decode_bits(
    encoded_bits: np.ndarray,
    original_bit_length: int,
    bch,
) -> np.ndarray:
    """
    Décode et corrige les bits du watermark avec BCH.

    decode() détecte et localise les erreurs.
    correct() applique ensuite la correction aux données.

    original_bit_length permet de supprimer le padding
    éventuellement ajouté lors de la conversion bits -> bytes.
    """

    if len(encoded_bits) == 0:
        return np.array(
            [],
            dtype=np.uint8,
        )

    encoded = bits_to_bytes(
        encoded_bits
    )

    block_size = (
        (bch.n // 8)
        - bch.ecc_bytes
    )

    encoded_block_size = (
        block_size
        + bch.ecc_bytes
    )

    decoded = bytearray()

    total_errors = 0
    failed_blocks = 0
    processed_blocks = 0

    for start in range(
        0,
        len(encoded),
        encoded_block_size,
    ):

        block = encoded[
            start:start + encoded_block_size
        ]

        # Un bloc incomplet ne contenant pas
        # au moins un octet de données.
        if len(block) <= bch.ecc_bytes:
            break

        data_len = (
            len(block)
            - bch.ecc_bytes
        )

        data = bytearray(
            block[:data_len]
        )

        ecc = bytearray(
            block[data_len:]
        )

        processed_blocks += 1

        try:

            bitflips = bch.decode(
                data=data,
                recv_ecc=ecc,
            )

            if bitflips < 0:

                failed_blocks += 1

                logger.warning(
                    "bch_decode_failed",
                    block=int(processed_blocks),
                    bitflips=int(bitflips),
                )

                # On conserve les données non corrigées.
                decoded.extend(data)

            else:

                # IMPORTANT :
                # decode() ne modifie pas nécessairement
                # les données.
                #
                # correct() applique la correction.
                bch.correct(
                    data=data,
                    ecc=ecc,
                )

                total_errors += int(
                    bitflips
                )

                decoded.extend(data)

        except (
            ValueError,
            IndexError,
        ) as exc:

            failed_blocks += 1

            logger.warning(
                "bch_decode_invalid_block",
                block=int(processed_blocks),
                error=str(exc),
            )

            decoded.extend(data)

    decoded_bits = bytes_to_bits(
        bytes(decoded)
    )

    result = decoded_bits[
        :original_bit_length
    ]

    logger.info(
        "bch_watermark_decoded",
        blocks=int(processed_blocks),
        total_errors=int(total_errors),
        failed_blocks=int(failed_blocks),
        original_bits=int(
            original_bit_length
        ),
    )

    return result


# ============================================================
# CHEMINS
# ============================================================

def _module_dir() -> str:
    """
    Retourne le dossier du module courant.
    """
    return os.path.dirname(
        os.path.abspath(__file__)
    )


def _resolve_logo_path(
    logo_filename: str,
) -> str:
    """
    Résout le chemin du logo.
    """
    return os.path.join(
        _module_dir(),
        logo_filename,
    )


# ============================================================
# UTILITAIRES IMAGE
# ============================================================

def _pad_to_power_of_2(
    arr: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Complète un tableau jusqu'à des dimensions
    puissances de deux.
    """

    h, w = arr.shape[:2]

    new_h = 1 << (
        h - 1
    ).bit_length()

    new_w = 1 << (
        w - 1
    ).bit_length()

    if arr.ndim == 3:

        padded = np.zeros(
            (
                new_h,
                new_w,
                arr.shape[2],
            ),
            dtype=arr.dtype,
        )

    else:

        padded = np.zeros(
            (
                new_h,
                new_w,
            ),
            dtype=arr.dtype,
        )

    padded[
        :h,
        :w,
    ] = arr

    return padded, (h, w)


# ============================================================
# FWHT
# ============================================================

def fwht_1d(
    x: np.ndarray,
) -> np.ndarray:
    """
    Fast Walsh-Hadamard Transform 1D.
    """

    n = len(x)

    result = (
        x.astype(np.float64)
        .copy()
    )

    h = 1

    while h < n:

        for i in range(
            0,
            n,
            h * 2,
        ):

            for j in range(
                i,
                i + h,
            ):

                a = result[j]
                b = result[j + h]

                result[j] = (
                    a + b
                )

                result[j + h] = (
                    a - b
                )

        h *= 2

    return result / n


def ifwht_1d(
    x: np.ndarray,
) -> np.ndarray:
    """
    Inverse Fast Walsh-Hadamard Transform 1D.
    """

    n = len(x)

    result = (
        x.astype(np.float64)
        .copy()
    )

    h = 1

    while h < n:

        for i in range(
            0,
            n,
            h * 2,
        ):

            for j in range(
                i,
                i + h,
            ):

                a = result[j]
                b = result[j + h]

                result[j] = (
                    a + b
                )

                result[j + h] = (
                    a - b
                )

        h *= 2

    return result


def fwht_2d(
    block: np.ndarray,
) -> np.ndarray:
    """
    FWHT 2D.
    """

    rows = np.array(
        [
            fwht_1d(row)
            for row in block
        ]
    )

    cols = np.array(
        [
            fwht_1d(col)
            for col in rows.T
        ]
    ).T

    return cols


def ifwht_2d(
    block: np.ndarray,
) -> np.ndarray:
    """
    Inverse FWHT 2D.
    """

    rows = np.array(
        [
            ifwht_1d(row)
            for row in block
        ]
    )

    cols = np.array(
        [
            ifwht_1d(col)
            for col in rows.T
        ]
    ).T

    return cols


# ============================================================
# PREPARATION DU LOGO
# ============================================================

def _prepare_logo_bits(
    logo_filename: str,
    target_shape: tuple[int, int],
    threshold: int = 127,
    max_ratio: float = 0.25,
) -> tuple[np.ndarray, dict]:

    logo_path = _resolve_logo_path(
        logo_filename
    )

    if not os.path.exists(logo_path):
        raise FileNotFoundError(
            f"Logo introuvable : {logo_path}"
        )

    logo = Image.open(
        logo_path
    ).convert("L")

    target_h, target_w = (
        target_shape
    )

    max_h = max(
        1,
        int(
            (target_h // 8)
            * max_ratio
        ),
    )

    max_w = max(
        1,
        int(
            (target_w // 8)
            * max_ratio
        ),
    )

    max_h = max(
        max_h,
        8,
    )

    max_w = max(
        max_w,
        8,
    )

    if logo.size != (
        max_w,
        max_h,
    ):

        logo = logo.resize(
            (
                max_w,
                max_h,
            ),
            Image.Resampling.LANCZOS,
        )

    logo_arr = np.array(
        logo,
        dtype=np.uint8,
    )

    bits = (
        logo_arr > threshold
    ).astype(np.uint8)

    meta = {
        "logo_filename": logo_filename,
        "logo_path": logo_path,
        "logo_shape": [
            int(bits.shape[0]),
            int(bits.shape[1]),
        ],
        "logo_bits": int(
            bits.size
        ),
        "logo_sha256": hashlib.sha256(
            logo_arr.tobytes()
        ).hexdigest(),
    }

    return bits, meta


# ============================================================
# ROTATION
# ============================================================

def _rotate_90_candidates(
    arr: np.ndarray,
) -> list[np.ndarray]:
    """
    Génère les quatre orientations possibles.
    """

    return [
        np.rot90(
            arr,
            k=k,
            axes=(0, 1),
        ).copy()
        for k in range(4)
    ]


# ============================================================
# SIFT + RANSAC
# ============================================================

def _sift_homography_warp(
    reference_bgr: np.ndarray,
    target_bgr: np.ndarray,
) -> tuple[np.ndarray, dict]:

    if cv2 is None:

        return target_bgr, {
            "homography_found": False,
            "sift_matches": 0,
            "sift_inliers": 0,
        }

    ref_gray = cv2.cvtColor(
        reference_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    tgt_gray = cv2.cvtColor(
        target_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    sift = cv2.SIFT_create()

    kp1, des1 = (
        sift.detectAndCompute(
            ref_gray,
            None,
        )
    )

    kp2, des2 = (
        sift.detectAndCompute(
            tgt_gray,
            None,
        )
    )

    if (
        des1 is None
        or des2 is None
        or len(kp1) < 4
        or len(kp2) < 4
    ):

        return target_bgr, {
            "homography_found": False,
            "sift_matches": 0,
            "sift_inliers": 0,
        }

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(
        des1,
        des2,
        k=2,
    )

    good = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if (
            m.distance
            < 0.75 * n.distance
        ):
            good.append(m)

    if len(good) < 4:

        return target_bgr, {
            "homography_found": False,
            "sift_matches": len(good),
            "sift_inliers": 0,
        }

    src_pts = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in good
        ]
    ).reshape(
        -1,
        1,
        2,
    )

    dst_pts = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in good
        ]
    ).reshape(
        -1,
        1,
        2,
    )

    H, mask = cv2.findHomography(
        dst_pts,
        src_pts,
        cv2.RANSAC,
        5.0,
    )

    if H is None:

        return target_bgr, {
            "homography_found": False,
            "sift_matches": len(good),
            "sift_inliers": 0,
        }

    inliers = (
        int(mask.ravel().sum())
        if mask is not None
        else 0
    )

    warped = cv2.warpPerspective(
        target_bgr,
        H,
        (
            reference_bgr.shape[1],
            reference_bgr.shape[0],
        ),
    )

    return warped, {
        "homography_found": True,
        "sift_matches": len(good),
        "sift_inliers": inliers,
    }


# ============================================================
# PLACEMENT DES BITS
# ============================================================

def _select_logo_blocks(
    image_shape: tuple[int, int],
    logo_shape: tuple[int, int],
    block_size: int,
    redundancy: int,
) -> list[list[tuple[int, int]]]:

    h, w = image_shape

    lh, lw = logo_shape

    bh = h // block_size
    bw = w // block_size

    if lh > bh or lw > bw:

        raise ValueError(
            "Le watermark est trop grand "
            "pour le nombre de blocs disponible."
        )

    total_bits = (
        lh * lw
    )

    total_blocks = (
        bh * bw
    )

    needed = (
        total_bits
        * redundancy
    )

    if needed > total_blocks:

        raise ValueError(
            "Image trop petite pour contenir "
            f"{total_bits} bits × "
            f"{redundancy} répétitions. "
            f"Blocs disponibles={total_blocks}, "
            f"blocs nécessaires={needed}."
        )

    # --------------------------------------------------------
    # Positions de tous les blocs
    # --------------------------------------------------------

    all_blocks = [
        (i, j)
        for i in range(bh)
        for j in range(bw)
    ]

    # --------------------------------------------------------
    # On évite de réutiliser immédiatement
    # les mêmes blocs.
    # --------------------------------------------------------

    placements = []

    cursor = 0

    for bit_idx in range(
        total_bits
    ):

        bit_blocks = []

        for replica in range(
            redundancy
        ):

            if cursor >= len(
                all_blocks
            ):
                raise ValueError(
                    "Plus assez de blocs disponibles."
                )

            block = all_blocks[
                cursor
            ]

            bit_blocks.append(
                block
            )

            cursor += 1

        placements.append(
            bit_blocks
        )

    return placements


# ============================================================
# INSERTION D'UN BIT FWHT
# ============================================================

def _embed_bit_in_block(
    watermarked: np.ndarray,
    block_i: int,
    block_j: int,
    bit: float,
    alpha: float,
    block_size: int,
) -> None:

    r_start = (
        block_i
        * block_size
    )

    r_end = (
        (block_i + 1)
        * block_size
    )

    c_start = (
        block_j
        * block_size
    )

    c_end = (
        (block_j + 1)
        * block_size
    )

    wm_val = (
        (2 * bit - 1)
        * alpha
    )

    for ch in range(
        watermarked.shape[2]
    ):

        block = watermarked[
            r_start:r_end,
            c_start:c_end,
            ch,
        ]

        padded, orig_shape = (
            _pad_to_power_of_2(
                block
            )
        )

        coeffs = fwht_2d(
            padded
        )

        mid = block_size // 2

        dc = max(
            abs(
                coeffs[0, 0]
            ),
            1.0,
        )

        coeffs[
            mid,
            mid
        ] += (
            wm_val
            * dc
        )

        second_row = min(
            mid + 1,
            block_size - 1,
        )

        coeffs[
            second_row,
            mid
        ] += (
            wm_val
            * dc
            * 0.5
        )

        reconstructed = (
            ifwht_2d(
                coeffs
            )
        )

        watermarked[
            r_start:r_end,
            c_start:c_end,
            ch,
        ] = reconstructed[
            :orig_shape[0],
            :orig_shape[1],
        ]


# ============================================================
# LECTURE D'UN BIT FWHT
# ============================================================

def _read_bit_score_from_block(
    candidate: np.ndarray,
    reference: np.ndarray,
    block_i: int,
    block_j: int,
    block_size: int,
) -> float:

    r_start = (
        block_i
        * block_size
    )

    r_end = (
        (block_i + 1)
        * block_size
    )

    c_start = (
        block_j
        * block_size
    )

    c_end = (
        (block_j + 1)
        * block_size
    )

    if (
        r_end > candidate.shape[0]
        or c_end > candidate.shape[1]
    ):
        return 0.0

    score = 0.0

    mid = block_size // 2

    second_row = min(
        mid + 1,
        block_size - 1,
    )

    for ch in range(
        candidate.shape[2]
    ):

        ref_block = reference[
            r_start:r_end,
            c_start:c_end,
            ch,
        ]

        cand_block = candidate[
            r_start:r_end,
            c_start:c_end,
            ch,
        ]

        ref_padded = (
            _pad_to_power_of_2(
                ref_block
            )[0]
        )

        cand_padded = (
            _pad_to_power_of_2(
                cand_block
            )[0]
        )

        coeffs_ref = fwht_2d(
            ref_padded
        )

        coeffs_cand = fwht_2d(
            cand_padded
        )

        score += (
            coeffs_cand[
                mid,
                mid
            ]
            -
            coeffs_ref[
                mid,
                mid
            ]
        )

        score += 0.5 * (
            coeffs_cand[
                second_row,
                mid
            ]
            -
            coeffs_ref[
                second_row,
                mid
            ]
        )

    return float(score)


# ============================================================
# EMBEDDING DU WATERMARK
# ============================================================

def embed_logo_watermark(
    image_bytes: bytes,
    logo_filename: str = "Blason_univ_Yaoundé_1.png",
    alpha: float = 0.005,
    block_size: int = 8,
    redundancy: int = 3,
) -> tuple[bytes, dict]:

    if block_size < 2:

        raise ValueError(
            "block_size doit être >= 2."
        )

    if (
        block_size
        & (block_size - 1)
    ):

        raise ValueError(
            "block_size doit être "
            "une puissance de 2."
        )

    if redundancy < 1:

        raise ValueError(
            "redundancy doit être >= 1."
        )

    # --------------------------------------------------------
    # Image originale
    # --------------------------------------------------------

    img = Image.open(
        io.BytesIO(
            image_bytes
        )
    ).convert("RGB")

    original = np.array(
        img,
        dtype=np.float64,
    )

    watermarked = (
        original.copy()
    )

    # --------------------------------------------------------
    # Logo
    # --------------------------------------------------------

    logo_bits, logo_meta = (
        _prepare_logo_bits(
            logo_filename,
            original.shape[:2],
        )
    )

    raw_bits = (
        logo_bits
        .flatten()
        .astype(np.uint8)
    )

    # --------------------------------------------------------
    # BCH
    # --------------------------------------------------------

    bch = create_bch_codec()

    encoded_bits = (
        bch_encode_bits(
            raw_bits,
            bch,
        )
    )

    if len(encoded_bits) == 0:

        raise ValueError(
            "Le watermark BCH est vide."
        )

    # --------------------------------------------------------
    # Vérification capacité
    #
    # On considère le watermark BCH comme une ligne
    # de bits : (1, nombre_de_bits).
    # --------------------------------------------------------

    encoded_shape = (
        1,
        len(encoded_bits),
    )

    placements = (
        _select_logo_blocks(
            original.shape[:2],
            encoded_shape,
            block_size,
            redundancy,
        )
    )

    # --------------------------------------------------------
    # Insertion FWHT
    # --------------------------------------------------------

    for bit_idx, bit in enumerate(
        encoded_bits
    ):

        for block_i, block_j in (
            placements[bit_idx]
        ):

            _embed_bit_in_block(
                watermarked,
                block_i,
                block_j,
                float(bit),
                alpha,
                block_size,
            )

    # --------------------------------------------------------
    # Conversion uint8
    # --------------------------------------------------------

    watermarked = np.clip(
        np.rint(
            watermarked
        ),
        0,
        255,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = np.mean(
        (
            original
            -
            watermarked.astype(
                np.float64
            )
        ) ** 2
    )

    # --------------------------------------------------------
    # PSNR
    # --------------------------------------------------------

    psnr = (
        10
        * np.log10(
            255.0 ** 2
            / max(
                mse,
                1e-10,
            )
        )
    )

    # --------------------------------------------------------
    # NC
    # --------------------------------------------------------

    orig_flat = (
        original.flatten()
    )

    wm_flat = (
        watermarked.astype(
            np.float64
        ).flatten()
    )

    denominator = (
        np.linalg.norm(
            orig_flat
        )
        *
        np.linalg.norm(
            wm_flat
        )
    )

    if denominator > 0:

        nc = (
            np.dot(
                orig_flat,
                wm_flat,
            )
            /
            denominator
        )

    else:

        nc = 0.0

    # --------------------------------------------------------
    # Sauvegarde PNG
    # --------------------------------------------------------

    buf = io.BytesIO()

    Image.fromarray(
        watermarked
    ).save(
        buf,
        format="PNG",
        optimize=True,
    )

    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------

    metrics = {

        "mode":
            "fwht_logo_bch_sift_ransac_rot90",

        # Qualité image
        "psnr_db":
            round(
                float(psnr),
                2,
            ),

        "nc":
            round(
                float(nc),
                6,
            ),

        "mse":
            round(
                float(mse),
                6,
            ),

        # Paramètres
        "alpha":
            alpha,

        "block_size":
            block_size,

        "redundancy":
            redundancy,

        # BCH
        "bch_t":
            BCH_T,

        "bch_m":
            BCH_M,

        "bch_n":
            int(bch.n),

        "bch_ecc_bytes":
            int(bch.ecc_bytes),

        "bch_ecc_bits":
            int(bch.ecc_bits),

        # Taille watermark
        "raw_logo_bits":
            int(len(raw_bits)),

        "bch_encoded_bits":
            int(len(encoded_bits)),

        "embedded_bits":
            int(
                len(encoded_bits)
                * redundancy
            ),

        **logo_meta,
    }

    logger.info(
        "fwht_logo_bch_watermark_embedded",
        **metrics,
    )

    return (
        buf.getvalue(),
        metrics,
    )


# ============================================================
# EXTRACTION D'UNE ORIENTATION
# ============================================================

def _extract_once(
    attacked: np.ndarray,
    original: np.ndarray,
    logo_bits: np.ndarray,
    logo_meta: dict,
    block_size: int,
    redundancy: int,
    alpha: float,
) -> tuple[np.ndarray, dict]:

    bch = create_bch_codec()

    # --------------------------------------------------------
    # Bits originaux du logo
    # --------------------------------------------------------

    expected_raw_bits = (
        logo_bits
        .flatten()
        .astype(np.uint8)
    )

    original_bit_length = (
        len(expected_raw_bits)
    )

    # --------------------------------------------------------
    # Recalcul du watermark BCH attendu
    # --------------------------------------------------------

    reference_encoded_bits = (
        bch_encode_bits(
            expected_raw_bits,
            bch,
        )
    )

    encoded_bit_length = (
        len(reference_encoded_bits)
    )

    if encoded_bit_length == 0:

        raise ValueError(
            "Le watermark BCH attendu est vide."
        )

    # --------------------------------------------------------
    # Positions des blocs
    # --------------------------------------------------------

    encoded_shape = (
        1,
        encoded_bit_length,
    )

    placements = (
        _select_logo_blocks(
            original.shape[:2],
            encoded_shape,
            block_size,
            redundancy,
        )
    )

    extracted_encoded_bits = (
        np.zeros(
            encoded_bit_length,
            dtype=np.uint8,
        )
    )

    # --------------------------------------------------------
    # Extraction FWHT
    # --------------------------------------------------------

    for idx in range(
        encoded_bit_length
    ):

        scores = []

        for block_i, block_j in (
            placements[idx]
        ):

            score = (
                _read_bit_score_from_block(
                    attacked,
                    original,
                    block_i,
                    block_j,
                    block_size,
                )
            )

            scores.append(score)

        if not scores:

            extracted_encoded_bits[
                idx
            ] = 0

            continue

        # ----------------------------------------------------
        # Vote majoritaire
        # ----------------------------------------------------

        positive_votes = sum(
            score > 0
            for score in scores
        )

        extracted_encoded_bits[
            idx
        ] = (
            1
            if positive_votes
            >= (
                len(scores)
                / 2
            )
            else 0
        )

    # --------------------------------------------------------
    # BCH : correction
    # --------------------------------------------------------

    corrected_bits = (
        bch_decode_bits(
            extracted_encoded_bits,
            original_bit_length,
            bch,
        )
    )

    # --------------------------------------------------------
    # Sécurité sur la longueur
    # --------------------------------------------------------

    if len(corrected_bits) != (
        len(expected_raw_bits)
    ):

        padded = np.zeros(
            len(expected_raw_bits),
            dtype=np.uint8,
        )

        n = min(
            len(corrected_bits),
            len(padded),
        )

        if n > 0:

            padded[:n] = (
                corrected_bits[:n]
            )

        corrected_bits = padded

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    bit_accuracy = float(
        np.mean(
            corrected_bits
            ==
            expected_raw_bits
        )
    )

    # --------------------------------------------------------
    # Corrélation
    # --------------------------------------------------------

    if len(expected_raw_bits) > 1:

        corr = np.corrcoef(
            expected_raw_bits,
            corrected_bits,
        )[0, 1]

    else:

        corr = 0.0

    if np.isnan(corr):

        corr = 0.0

    corr = float(corr)

    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------

    metrics = {

        "correlation":
            round(
                corr,
                6,
            ),

        "bit_accuracy":
            round(
                bit_accuracy,
                6,
            ),

        "alpha":
            alpha,

        "block_size":
            block_size,

        "redundancy":
            redundancy,

        "logo_bits":
            int(
                original_bit_length
            ),

        "bch_encoded_bits":
            int(
                encoded_bit_length
            ),

        "bch_t":
            BCH_T,

        "bch_m":
            BCH_M,

        **logo_meta,
    }

    return (
        corrected_bits.astype(
            np.float64
        ),
        metrics,
    )


# ============================================================
# EXTRACTION COMPLETE
# ============================================================

def extract_logo_watermark(
    attacked_bytes: bytes,
    original_bytes: bytes,
    logo_filename: str = "Blason_univ_Yaoundé_1.png",
    alpha: float = 0.005,
    block_size: int = 8,
    redundancy: int = 3,
) -> tuple[np.ndarray, dict]:

    # --------------------------------------------------------
    # Chargement image attaquée
    # --------------------------------------------------------

    attacked_rgb = np.array(
        Image.open(
            io.BytesIO(
                attacked_bytes
            )
        ).convert("RGB"),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Chargement image originale
    # --------------------------------------------------------

    original_rgb = np.array(
        Image.open(
            io.BytesIO(
                original_bytes
            )
        ).convert("RGB"),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Logo
    # --------------------------------------------------------

    logo_bits, logo_meta = (
        _prepare_logo_bits(
            logo_filename,
            original_rgb.shape[:2],
        )
    )

    # --------------------------------------------------------
    # Meilleur résultat
    # --------------------------------------------------------

    best_bits = None
    best_metrics = None

    best_score = -1.0

    best_rot = 0

    best_geom = {
        "homography_found": False,
        "sift_matches": 0,
        "sift_inliers": 0,
    }

    # --------------------------------------------------------
    # Tester les 4 rotations
    # --------------------------------------------------------

    for k in range(4):

        candidate = np.rot90(
            attacked_rgb,
            k=k,
            axes=(0, 1),
        ).copy()

        # ----------------------------------------------------
        # Alignement SIFT + RANSAC
        # ----------------------------------------------------

        aligned, geom_metrics = (
            _sift_homography_warp(
                original_rgb,
                candidate,
            )
        )

        # ----------------------------------------------------
        # Extraction
        # ----------------------------------------------------

        bits, metrics = (
            _extract_once(
                aligned.astype(
                    np.float64
                ),
                original_rgb.astype(
                    np.float64
                ),
                logo_bits,
                logo_meta,
                block_size,
                redundancy,
                alpha,
            )
        )

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score = (
            metrics["bit_accuracy"]
            +
            max(
                metrics["correlation"],
                0.0,
            )
        )

        if score > best_score:

            best_score = score

            best_bits = bits

            best_metrics = metrics

            best_rot = k

            best_geom = (
                geom_metrics
            )

    # --------------------------------------------------------
    # Vérification
    # --------------------------------------------------------

    if best_bits is None:

        raise RuntimeError(
            "Aucun watermark n'a pu être extrait."
        )

    if best_metrics is None:

        raise RuntimeError(
            "Les métriques d'extraction "
            "sont indisponibles."
        )

    # --------------------------------------------------------
    # Métriques géométriques
    # --------------------------------------------------------

    best_metrics.update(
        best_geom
    )

    best_metrics[
        "rotation_k"
    ] = best_rot

    best_metrics[
        "mode"
    ] = (
        "fwht_logo_bch_sift_ransac_rot90_extract"
    )

    best_metrics[
        "best_score"
    ] = round(
        float(best_score),
        6,
    )

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    logger.info(
        "fwht_logo_bch_watermark_extracted",
        **best_metrics,
    )

    return (
        best_bits,
        best_metrics,
    )
