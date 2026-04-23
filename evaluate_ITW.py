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

    # Use all unique scores as candidate thresholds
    all_scores = np.sort(np.unique(scores))

    # Add two boundary thresholds to complete the curve
    thresholds = np.concatenate((
        [all_scores[0] - 1e-6],
        all_scores,
        [all_scores[-1] + 1e-6]
    ))

    fars = []   # False Acceptance Rate: spoof classified as bonafide
    frrs = []   # False Rejection Rate: bonafide classified as spoof

    for thr in thresholds:
        # Predict bonafide if score >= threshold
        far = np.mean(neg_scores >= thr)
        frr = np.mean(pos_scores < thr)
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars, dtype=np.float64)
    frrs = np.array(frrs, dtype=np.float64)

    diff = fars - frrs
    idx = np.argmin(np.abs(diff))

    # Directly use the closest point if it is at the boundary
    if idx == 0 or idx == len(thresholds) - 1:
        eer = (fars[idx] + frrs[idx]) / 2.0
        return float(eer), float(thresholds[idx])

    # If FAR and FRR are exactly equal
    if diff[idx] == 0:
        eer = fars[idx]
        thr = thresholds[idx]
        return float(eer), float(thr)

    # Find the interval where the sign changes
    i0 = None
    for k in range(len(diff) - 1):
        if diff[k] == 0:
            return float(fars[k]), float(thresholds[k])
        if diff[k] * diff[k + 1] < 0:
            i0 = k
            break

    # If no sign change is found, use the closest point
    if i0 is None:
        eer = (fars[idx] + frrs[idx]) / 2.0
        return float(eer), float(thresholds[idx])

    x0 = thresholds[i0]
    x1 = thresholds[i0 + 1]
    d0 = diff[i0]
    d1 = diff[i0 + 1]

    # Linear interpolation to estimate the threshold where FAR == FRR
    thr = x0 - d0 * (x1 - x0) / (d1 - d0 + 1e-12)

    far0, far1 = fars[i0], fars[i0 + 1]
    frr0, frr1 = frrs[i0], frrs[i0 + 1]

    w = (thr - x0) / (x1 - x0 + 1e-12)
    far = far0 + w * (far1 - far0)
    frr = frr0 + w * (frr1 - frr0)
    eer = (far + frr) / 2.0

    return float(eer), float(thr)


def main():
    parser = argparse.ArgumentParser(description="Compute EER for ITW score file and protocol file.")
    parser.add_argument(
        "--score_file",
        type=str,
        required=True,
        help="Path to the score file, which must contain columns: filename, cm-score"
    )
    parser.add_argument(
        "--label_file",
        type=str,
        required=True,
        help="Path to the protocol/label file, which must contain columns: filename, cm_label"
    )
    parser.add_argument(
        "--save_merged",
        type=str,
        default="",
        help="Optional path to save the merged details, e.g., merged_itw.csv"
    )
    args = parser.parse_args()

    # Read the input files
    score_df = read_table_auto(args.score_file)
    label_df = read_table_auto(args.label_file)

    # Check required columns
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

    # Normalize file paths
    score_df["filename"] = score_df["filename"].astype(str).str.strip().str.replace("\\", "/", regex=False)
    label_df["filename"] = label_df["filename"].astype(str).str.strip().str.replace("\\", "/", regex=False)

    # First try exact path matching
    merged_exact = pd.merge(
        score_df[["filename", "cm-score"]],
        label_df[["filename", "cm_label"]],
        on="filename",
        how="inner"
    )

    # Then try basename matching and keep the better one
    score_df["basename"] = score_df["filename"].apply(os.path.basename)
    label_df["basename"] = label_df["filename"].apply(os.path.basename)

    merged_base = pd.merge(
        score_df[["filename", "basename", "cm-score"]],
        label_df[["filename", "basename", "cm_label"]],
        on="basename",
        how="inner",
        suffixes=("_score", "_label")
    )

    if len(merged_base) > len(merged_exact):
        merged = merged_base.rename(columns={"filename_score": "filename"})
        merge_mode = "basename"
    else:
        merged = merged_exact.copy()
        merge_mode = "exact filename"

    if len(merged) == 0:
        raise ValueError("No matched utterances were found between the score file and the label file.")

    # Clean and validate labels
    merged["cm_label"] = merged["cm_label"].astype(str).str.strip().str.lower()
    valid_labels = {"bonafide", "spoof"}
    found_labels = set(merged["cm_label"].unique().tolist())

    if not found_labels.issubset(valid_labels):
        raise ValueError(f"Invalid labels found: {found_labels}. Labels must be bonafide/spoof.")

    merged["label"] = merged["cm_label"].map({"bonafide": 1, "spoof": 0})
    merged["cm-score"] = merged["cm-score"].astype(float)

    bonafide_num = int((merged["label"] == 1).sum())
    spoof_num = int((merged["label"] == 0).sum())

    # Automatically check whether score direction needs to be reversed
    bona_mean = merged.loc[merged["label"] == 1, "cm-score"].mean()
    spoof_mean = merged.loc[merged["label"] == 0, "cm-score"].mean()

    scores = merged["cm-score"].values.copy()
    if bona_mean < spoof_mean:
        # Reverse scores if bonafide scores are smaller on average
        scores = -scores
        score_direction = "reversed automatically"
    else:
        score_direction = "original"

    eer, threshold = compute_eer(scores, merged["label"].values)

    print("=" * 60)
    print("ITW EER Evaluation Result")
    print("=" * 60)
    print(f"EER                  : {eer * 100:.4f}%")
    print(f"Threshold            : {threshold:.10f}")
    print("=" * 60)

    if args.save_merged:
        out_df = merged.copy()
        out_df["used_score"] = scores
        out_df.to_csv(args.save_merged, index=False, encoding="utf-8-sig")
        print(f"Merged details saved to: {args.save_merged}")


if __name__ == "__main__":
    main()