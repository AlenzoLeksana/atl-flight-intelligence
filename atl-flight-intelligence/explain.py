"""SHAP lokal pada model yang sama dengan aplikasi (Baseline V1)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import shap


def local_waterfall(bundle, X):
    model = bundle["model"]
    # Output raw (log-odds), bukan perubahan langsung dalam probabilitas.
    explainer = shap.TreeExplainer(model.booster_, model_output="raw")
    exp = explainer(X)
    values = np.asarray(exp.values)
    base = np.asarray(exp.base_values)
    if values.ndim != 2 or values.shape != (1, X.shape[1]):
        raise ValueError(f"Bentuk SHAP {values.shape} perlu diaudit sebelum ditampilkan.")
    if base.size != 1:
        raise ValueError(f"Bentuk base value {base.shape} perlu diaudit.")
    raw = float(np.asarray(model.booster_.predict(X, raw_score=True)).reshape(-1)[0])
    if not np.isclose(float(base.reshape(-1)[0]) + values[0].sum(), raw,
                      rtol=1e-5, atol=1e-5):
        raise AssertionError("Sifat aditif SHAP tidak cocok dengan raw score model.")
    shap.plots.waterfall(exp[0], max_display=X.shape[1], show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    return fig, raw
