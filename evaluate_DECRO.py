#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd


def read_table_auto(path):
    """
    Automatically detect the delimiter and read a txt/csv/tsv file.
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {path}\n{e}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def compute_eer(scores, labels):
    """
    Compute Equal Error Rate (EER).

    Convention:
      labels = 1 means bonafide
      labels = 0 means spoof
      larger score means more likely to be bonafide

    Returns:
      eer, threshold
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    pos_scores = scores[labels == 1]  # bonafide scores
    neg_scores = scores[labels == 0]  # spoof scores

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise ValueError("The number of bonafide or spoof samples is 0, cannot compute EER.")

    all_scores = np.sort(np.unique(scores))

    thresholds = np.concatenate((
        [all_scores[0] - 1e-6],
        all_scores,
        [all_scores[-1] + 1e-6]
    ))

    fars = []
    frrs = []

    for thr in thresholds:
        # Predict bonafide if score >= threshold
        far = np.mean(neg_scores >= thr)   # spoof classified as bonafide
        frr = np.mean(pos_scores < thr)    # bonafide classified as spoof
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars, dtype=np.float64)
    frrs = np.array(frrs, dtype=np.float64)

    diff = fars - frrs
    idx = np.argmin(np.abs(diff))

    if idx == 0 or idx == len(thresholds) - 1:
        eer = (fars[idx] + frrs[idx]) / 2.0
        return float(eer), float(thresholds[idx])

    if diff[idx] == 0:
        return float(fars[idx]), float(thresholds[idx])

    i0 = None
    for k in range(len(diff) - 1):
        if diff[k] == 0:
            return float(fars[k]), float(thresholds[k])
        if diff[k] * diff[k + 1] < 0:
            i0 = k
            break

    if i0 is None:
        eer = (fars[idx] + frrs[idx]) / 2.0
        return float(eer), float(thresholds[idx])

    x0 = thresholds[i0]
    x1 = thresholds[i0 + 1]
    d0 = diff[i0]
    d1 = diff[i0 + 1]

    thr = x0 - d0 * (x1 - x0) / (d1 - d0 + 1e-12)

    far0, far1 = fars[i0], fars[i0 + 1]
    frr0, frr1 = frrs[i0], frrs[i0 + 1]

    w = (thr - x0) / (x1 - x0 + 1e-12)
    far = far0 + w * (far1 - far0)
    frr = frr0 + w * (frr1 - frr0)
    eer = (far + frr) / 2.0

    return float(eer), float(thr)


def evaluate_one_set(score_file, label_file, save_merged=""):
    """
    Evaluate one score/protocol pair and return EER information.
    """
    score_df = read_table_auto(score_file)
    label_df = read_table_auto(label_file)

    required_score_cols = {"filename", "cm-score"}
    required_label_cols = {"filename", "cm_label"}

    if not required_score_cols.issubset(set(score_df.columns)):
        raise ValueError(
            f"Score file is missing required columns. Current columns: {list(score_df.columns)}. "
            f"Required: {list(required_score_cols)}"
        )

    if not required_label_cols.issubset(set(label_df.columns)):
        raise ValueError(
            f"Label file is missing required columns. Current columns: {list(label_df.columns)}. "
            f"Required: {list(required_label_cols)}"
        )

    score_df["filename"] = score_df["filename"].astype(str).str.strip().str.replace("\\", "/", regex=False)
    label_df["filename"] = label_df["filename"].astype(str).str.strip().str.replace("\\", "/", regex=False)

    merged = pd.merge(
        score_df[["filename", "cm-score"]],
        label_df[["filename", "cm_label"]],
        on="filename",
        how="inner"
    )

    if len(merged) == 0:
        raise ValueError(
            f"No matched utterances were found between:\n"
            f"  score_file = {score_file}\n"
            f"  label_file = {label_file}"
        )

    merged["cm_label"] = merged["cm_label"].astype(str).str.strip().str.lower()
    valid_labels = {"bonafide", "spoof"}
    found_labels = set(merged["cm_label"].unique().tolist())

    if not found_labels.issubset(valid_labels):
        raise ValueError(f"Invalid labels found: {found_labels}. Labels must be bonafide/spoof.")

    merged["label"] = merged["cm_label"].map({"bonafide": 1, "spoof": 0})
    merged["cm-score"] = merged["cm-score"].astype(float)

    bonafide_num = int((merged["label"] == 1).sum())
    spoof_num = int((merged["label"] == 0).sum())

    bona_mean = merged.loc[merged["label"] == 1, "cm-score"].mean()
    spoof_mean = merged.loc[merged["label"] == 0, "cm-score"].mean()

    scores = merged["cm-score"].values.copy()
    if bona_mean < spoof_mean:
        scores = -scores
        score_direction = "reversed automatically"
    else:
        score_direction = "original"

    eer, threshold = compute_eer(scores, merged["label"].values)

    if save_merged:
        out_df = merged.copy()
        out_df["used_score"] = scores
        out_df.to_csv(save_merged, index=False, encoding="utf-8-sig")

    return {
        "score_file": score_file,
        "label_file": label_file,
        "num_samples": len(merged),
        "num_bonafide": bonafide_num,
        "num_spoof": spoof_num,
        "score_direction": score_direction,
        "threshold": threshold,
        "eer": eer,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute EER for DECRO Chinese and English, then report their average EER."
    )
    parser.add_argument("--cn_score_file", type=str, required=True,
                        help="Path to DECRO Chinese score file")
    parser.add_argument("--cn_label_file", type=str, required=True,
                        help="Path to DECRO Chinese protocol/label file")
    parser.add_argument("--en_score_file", type=str, required=True,
                        help="Path to DECRO English score file")
    parser.add_argument("--en_label_file", type=str, required=True,
                        help="Path to DECRO English protocol/label file")
    parser.add_argument("--save_cn_merged", type=str, default="",
                        help="Optional path to save the merged Chinese details")
    parser.add_argument("--save_en_merged", type=str, default="",
                        help="Optional path to save the merged English details")
    args = parser.parse_args()

    cn_result = evaluate_one_set(
        score_file=args.cn_score_file,
        label_file=args.cn_label_file,
        save_merged=args.save_cn_merged
    )

    en_result = evaluate_one_set(
        score_file=args.en_score_file,
        label_file=args.en_label_file,
        save_merged=args.save_en_merged
    )

    avg_eer = (cn_result["eer"] + en_result["eer"]) / 2.0

    print("=" * 70)
    print("DECRO Evaluation Result")
    print("=" * 70)

    print("[Chinese]")
    print(f"Threshold           : {cn_result['threshold']:.10f}")
    print(f"EER                 : {cn_result['eer'] * 100:.4f}%")
    print("-" * 70)

    print("[English]")
    print(f"Threshold           : {en_result['threshold']:.10f}")
    print(f"EER                 : {en_result['eer'] * 100:.4f}%")
    print("-" * 70)

    print("[AVG]")
    print(f"AVG EER         : {avg_eer * 100:.4f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()