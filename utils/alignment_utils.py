from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class AlignmentCandidate:
    bias: int
    score: float
    signal_corr: float
    event_corr: float
    peak_score: float
    overlap_ratio: float


@dataclass
class AlignmentResult:
    best_bias: int
    score: float
    confidence: float
    status: str
    candidates: list
    diagnostics: dict

    def to_dict(self):
        return asdict(self)


def robust_normalize(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if not finite.all():
        fill = np.nanmedian(x[finite]) if finite.any() else 0.0
        x = np.where(finite, x, fill)
    x = x - np.median(x)
    mad = np.median(np.abs(x - np.median(x)))
    scale = 1.4826 * mad
    if scale < eps:
        q75, q25 = np.percentile(x, [75, 25])
        scale = (q75 - q25) / 1.349
    if scale < eps:
        scale = np.std(x)
    if scale < eps:
        return np.zeros_like(x)
    return np.clip(x / scale, -8.0, 8.0)


def preprocess_signal(x, fps=30, smooth_sec=0.25):
    x = robust_normalize(x)
    win = int(round(smooth_sec * fps))
    if win >= 3 and x.size > win:
        kernel = np.ones(win, dtype=np.float64) / win
        x = np.convolve(x, kernel, mode="same")
    x = robust_normalize(x)
    event = np.abs(np.gradient(x))
    event = robust_normalize(event)
    return x, event


def crop_pair(sensor, mocap, bias):
    sensor = np.asarray(sensor)
    mocap = np.asarray(mocap)
    if bias > 0:
        sensor = sensor[bias:]
    elif bias < 0:
        mocap = mocap[-bias:]
    n = min(len(sensor), len(mocap))
    return sensor[:n], mocap[:n]


def crop_tensor_pair(sensor, mocap, bias):
    if bias > 0:
        sensor = sensor[bias:]
    elif bias < 0:
        mocap = mocap[-bias:]
    n = min(len(sensor), len(mocap))
    return sensor[:n], mocap[:n]


def _corr(a, b):
    if len(a) < 8 or len(b) < 8:
        return 0.0
    a = robust_normalize(a)
    b = robust_normalize(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _overlap_ratio(n_sensor, n_mocap, bias):
    if bias > 0:
        n = min(n_sensor - bias, n_mocap)
    elif bias < 0:
        n = min(n_sensor, n_mocap + bias)
    else:
        n = min(n_sensor, n_mocap)
    if n <= 0:
        return 0.0
    return float(n / min(n_sensor, n_mocap))


def _top_peaks(event, fps, max_peaks=40):
    if len(event) < fps:
        return np.array([], dtype=np.int64)
    distance = max(3, int(round(0.35 * fps)))
    threshold = max(0.2, np.percentile(event, 85) * 0.5)
    local = (event[1:-1] > event[:-2]) & (event[1:-1] >= event[2:]) & (event[1:-1] >= threshold)
    peaks = np.where(local)[0] + 1
    if peaks.size == 0:
        return peaks.astype(np.int64)
    strengths = event[peaks]
    order = np.argsort(strengths)[::-1]
    selected = []
    for idx in order:
        p = int(peaks[idx])
        if all(abs(p - q) >= distance for q in selected):
            selected.append(p)
        if len(selected) >= max_peaks:
            break
    return np.sort(np.asarray(selected, dtype=np.int64))


def _fft_correlate_full(a, b):
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b[::-1], size)
    return np.fft.irfft(fa * fb, size)[:n]


def _local_maxima(x, distance):
    x = np.asarray(x)
    if x.size < 3:
        return np.array([int(np.argmax(x))]) if x.size else np.array([], dtype=np.int64)
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if peaks.size == 0:
        return np.array([int(np.argmax(x))], dtype=np.int64)
    order = peaks[np.argsort(x[peaks])[::-1]]
    selected = []
    for p in order:
        p = int(p)
        if all(abs(p - q) >= distance for q in selected):
            selected.append(p)
    return np.asarray(selected, dtype=np.int64)


def _peak_match_score(sensor_peaks, mocap_peaks, bias, fps):
    if len(sensor_peaks) == 0 or len(mocap_peaks) == 0:
        return 0.0
    tol = max(2, int(round(0.12 * fps)))
    target = sensor_peaks - bias
    mocap_peaks = np.asarray(mocap_peaks)
    matched = 0
    errors = []
    for t in target:
        j = np.searchsorted(mocap_peaks, t)
        candidates = []
        if j < len(mocap_peaks):
            candidates.append(mocap_peaks[j])
        if j > 0:
            candidates.append(mocap_peaks[j - 1])
        if not candidates:
            continue
        err = min(abs(c - t) for c in candidates)
        if err <= tol:
            matched += 1
            errors.append(err)
    precision = matched / max(1, len(sensor_peaks))
    recall = matched / max(1, len(mocap_peaks))
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    if errors:
        jitter_penalty = min(0.25, float(np.std(errors)) / max(1, tol) * 0.1)
    else:
        jitter_penalty = 0.25
    return float(max(0.0, f1 - jitter_penalty))


def _combined_correlation_candidates(signal_pairs, fps, max_shift, top_k=12):
    combined = None
    lags_ref = None
    used = 0
    for pair in signal_pairs:
        s = pair["sensor_signal"]
        m = pair["mocap_signal"]
        weight = pair["weight"]
        if len(s) < fps or len(m) < fps or weight <= 0:
            continue
        corr = _fft_correlate_full(s, m)
        denom = np.linalg.norm(s) * np.linalg.norm(m)
        if denom < 1e-8:
            continue
        corr = corr / denom
        lags = np.arange(-len(m) + 1, len(s))
        mask = (lags >= -max_shift) & (lags <= max_shift)
        corr = corr[mask]
        lags = lags[mask]
        if combined is None:
            combined = np.zeros_like(corr, dtype=np.float64)
            lags_ref = lags
        if len(corr) == len(combined) and np.array_equal(lags, lags_ref):
            combined += weight * corr
            used += 1
    if combined is None or used == 0:
        return [0], None, None

    peaks = _local_maxima(combined, distance=max(3, int(0.5 * fps)))
    peak_order = peaks[np.argsort(combined[peaks])[::-1]]
    candidate_biases = [int(lags_ref[p]) for p in peak_order[:top_k]]
    if 0 not in candidate_biases:
        candidate_biases.append(0)
    return candidate_biases, lags_ref, combined


def _score_bias(signal_pairs, bias, fps):
    signal_corrs = []
    event_corrs = []
    peak_scores = []
    weights = []
    overlap_ratios = []
    for pair in signal_pairs:
        w = pair["weight"]
        ss, mm = crop_pair(pair["sensor_signal"], pair["mocap_signal"], bias)
        se, me = crop_pair(pair["sensor_event"], pair["mocap_event"], bias)
        overlap = _overlap_ratio(len(pair["sensor_signal"]), len(pair["mocap_signal"]), bias)
        if len(ss) < fps or overlap <= 0:
            continue
        signal_corrs.append(_corr(ss, mm))
        event_corrs.append(_corr(se, me))
        peak_scores.append(_peak_match_score(pair["sensor_peaks"], pair["mocap_peaks"], bias, fps))
        overlap_ratios.append(overlap)
        weights.append(w)

    if not weights:
        return AlignmentCandidate(int(bias), 0.0, 0.0, 0.0, 0.0, 0.0)

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    signal_corr = float(np.dot(weights, signal_corrs))
    event_corr = float(np.dot(weights, event_corrs))
    peak_score = float(np.dot(weights, peak_scores))
    overlap_ratio = float(np.dot(weights, overlap_ratios))
    positive_signal_corr = (signal_corr + 1.0) / 2.0
    positive_event_corr = (event_corr + 1.0) / 2.0
    score = (
        0.38 * positive_signal_corr
        + 0.28 * positive_event_corr
        + 0.24 * peak_score
        + 0.10 * overlap_ratio
    )
    return AlignmentCandidate(
        int(bias),
        float(score),
        signal_corr,
        event_corr,
        peak_score,
        overlap_ratio,
    )


def estimate_alignment_bias(sensor_signals, mocap_signals, fps=30, max_shift_sec=120.0):
    base_weights = {"left_wrist": 0.45, "right_wrist": 0.35, "head": 0.20}
    signal_pairs = []
    for name, base_weight in base_weights.items():
        if name not in sensor_signals or name not in mocap_signals:
            continue
        sensor_signal, sensor_event = preprocess_signal(sensor_signals[name], fps=fps)
        mocap_signal, mocap_event = preprocess_signal(mocap_signals[name], fps=fps)
        energy = min(np.std(sensor_signal), np.std(mocap_signal))
        if energy < 0.05:
            continue
        signal_pairs.append(
            {
                "name": name,
                "weight": base_weight * float(np.clip(energy, 0.05, 1.5)),
                "sensor_signal": sensor_signal,
                "mocap_signal": mocap_signal,
                "sensor_event": sensor_event,
                "mocap_event": mocap_event,
                "sensor_peaks": _top_peaks(sensor_event, fps=fps),
                "mocap_peaks": _top_peaks(mocap_event, fps=fps),
            }
        )

    if not signal_pairs:
        return AlignmentResult(
            best_bias=0,
            score=0.0,
            confidence=0.0,
            status="manual_required",
            candidates=[],
            diagnostics={"reason": "no_usable_signal_pairs"},
        )

    max_shift = int(round(max_shift_sec * fps))
    coarse_biases, lags, combined = _combined_correlation_candidates(signal_pairs, fps, max_shift)

    refine_biases = set()
    for b in coarse_biases:
        for delta in range(-10, 11):
            refine_biases.add(int(b + delta))
    refine_biases = [b for b in refine_biases if -max_shift <= b <= max_shift]

    candidates = [_score_bias(signal_pairs, b, fps) for b in refine_biases]
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

    best = candidates[0]
    separated = [c for c in candidates[1:] if abs(c.bias - best.bias) >= max(5, int(0.25 * fps))]
    second = separated[0] if separated else candidates[min(1, len(candidates) - 1)]
    margin = best.score - second.score
    quality = 0.50 * best.peak_score + 0.30 * max(0.0, best.event_corr) + 0.20 * best.overlap_ratio
    confidence = float(np.clip(0.55 * best.score + 1.4 * margin + 0.35 * quality, 0.0, 1.0))

    if confidence >= 0.62 and best.score >= 0.58 and best.peak_score >= 0.12:
        status = "auto_accept"
    elif confidence >= 0.48 and best.score >= 0.52:
        status = "review_recommended"
    else:
        status = "manual_required"

    diagnostics = {
        "used_channels": [p["name"] for p in signal_pairs],
        "channel_weights": {p["name"]: p["weight"] for p in signal_pairs},
        "second_best_bias": second.bias,
        "second_best_score": second.score,
        "margin": margin,
        "max_shift": max_shift,
    }
    if lags is not None and combined is not None:
        diagnostics["coarse_curve"] = {
            "lags": lags.astype(int).tolist(),
            "scores": combined.astype(float).tolist(),
        }

    return AlignmentResult(
        best_bias=best.bias,
        score=best.score,
        confidence=confidence,
        status=status,
        candidates=[asdict(c) for c in candidates[:10]],
        diagnostics=diagnostics,
    )


def save_alignment_report(report_dir, seq_name, result, sensor_signals, mocap_signals, fps=30):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    ordered = ["left_wrist", "head", "phone_right"]
    channels = [c for c in ordered if c in sensor_signals and c in mocap_signals]
    channels.extend(
        c for c in sensor_signals if c not in channels and c in mocap_signals
    )
    if not channels:
        return

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3.0 * len(channels)), squeeze=False)
    for ax, name in zip(axes[:, 0], channels):
        s, _ = preprocess_signal(sensor_signals[name], fps=fps)
        m, _ = preprocess_signal(mocap_signals[name], fps=fps)
        ss, mm = crop_pair(s, m, result.best_bias)
        n = min(len(ss), len(mm), fps * 45)
        ax.plot(ss[:n], label=f"sensor {name}", linewidth=1)
        ax.plot(mm[:n], label=f"mocap {name}", linewidth=1)
        ax.set_title(f"{seq_name} {name} bias={result.best_bias}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(report_dir / f"{seq_name}_overlay.png", dpi=140)
    plt.close(fig)

    curve = result.diagnostics.get("coarse_curve")
    if curve:
        lags = np.asarray(curve["lags"])
        scores = np.asarray(curve["scores"])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(lags, scores, linewidth=1)
        ax.axvline(result.best_bias, color="red", linestyle="--", label=f"best {result.best_bias}")
        ax.set_title(f"{seq_name} coarse correlation")
        ax.set_xlabel("bias frames")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(report_dir / f"{seq_name}_score_curve.png", dpi=140)
        plt.close(fig)
