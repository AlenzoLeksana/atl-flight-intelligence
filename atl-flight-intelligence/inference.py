"""Inferensi retrospektif LightGBM Baseline V1 menggunakan artefak 4.3.2."""
from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

EXPECTED_FEATURES = [
    "month", "day_of_week", "scheduled_dep_hour", "op_unique_carrier",
    "dest", "distance", "crs_elapsed_time"
]
CATEGORICAL = ("op_unique_carrier", "dest")


def verify_bundle(bundle: dict) -> None:
    required = {"scenario", "model", "feature_names", "categorical_features",
                "category_vocab_from_training", "threshold"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Kunci artefak model tidak lengkap: {sorted(missing)}")
    if bundle["scenario"] != "baseline":
        raise ValueError("Aplikasi dikunci untuk model Baseline V1, bukan Enhanced/V2.")
    if list(bundle["feature_names"]) != EXPECTED_FEATURES:
        raise ValueError("Urutan fitur artefak tidak sama dengan Baseline V1.")
    if set(bundle["categorical_features"]) != set(CATEGORICAL):
        raise ValueError("Metadata fitur kategorikal tidak konsisten.")
    if not 0 < float(bundle["threshold"]) < 1:
        raise ValueError("Threshold model tidak valid.")
    if bundle["model"].n_features_in_ != len(EXPECTED_FEATURES):
        raise ValueError("Dimensi fitur model tidak cocok.")
    for col in CATEGORICAL:
        if col not in bundle["category_vocab_from_training"]:
            raise ValueError(f"Kosakata kategori {col} tidak ditemukan.")


def make_features(flight: pd.Series, bundle: dict):
    """Rekonstruksi jadwal dari tanggal + HHMM, tanpa memuat informasi aktual target."""
    verify_bundle(bundle)
    date = pd.Timestamp(flight["fl_date"])
    if pd.isna(date) or not (date.year == 2024 and date.month in (11, 12)):
        raise ValueError("Pilihan harus penerbangan historis Nov–Des 2024.")
    hhmm = int(flight["crs_dep_time"])
    hh, minute = divmod(hhmm, 100)
    if not (0 <= hh < 24 and 0 <= minute < 60):
        raise ValueError("crs_dep_time harus berupa HHMM yang valid.")
    distance = float(flight["distance"])
    elapsed = float(flight["crs_elapsed_time"])
    if not (np.isfinite(distance) and distance > 0 and np.isfinite(elapsed) and elapsed > 0):
        raise ValueError("Jarak dan durasi terjadwal harus bernilai positif.")

    schedule = date.normalize() + pd.Timedelta(hours=hh, minutes=minute)
    cutoff = schedule - pd.Timedelta(hours=2)
    features = {
        "month": int(date.month),
        "day_of_week": int(date.dayofweek + 1),
        "scheduled_dep_hour": int(hh),
        "op_unique_carrier": str(flight["op_unique_carrier"]),
        "dest": str(flight["dest"]),
        "distance": distance,
        "crs_elapsed_time": elapsed,
    }
    X = pd.DataFrame([features], columns=EXPECTED_FEATURES)
    X["month"] = X["month"].astype("int64")
    X["day_of_week"] = X["day_of_week"].astype("int64")
    X["scheduled_dep_hour"] = X["scheduled_dep_hour"].astype("int8")
    X["distance"] = X["distance"].astype("float64")
    X["crs_elapsed_time"] = X["crs_elapsed_time"].astype("float64")

    unknown = []
    for col in CATEGORICAL:
        vocab = list(bundle["category_vocab_from_training"][col])
        if features[col] not in vocab:
            unknown.append((col, features[col]))
        X[col] = pd.Categorical(X[col], categories=vocab, ordered=False)
    return X, schedule, cutoff, unknown


def predict(flight: pd.Series, bundle: dict):
    X, schedule, cutoff, unknown = make_features(flight, bundle)
    prob = float(bundle["model"].predict_proba(X)[:, 1][0])
    if not np.isfinite(prob) or not 0 <= prob <= 1:
        raise AssertionError("Probabilitas model tidak valid.")
    threshold = float(bundle["threshold"])
    label = int(prob >= threshold)
    return {"prob": prob, "threshold": threshold, "label": label,
            "X": X, "schedule": schedule, "cutoff": cutoff, "unknown": unknown}
