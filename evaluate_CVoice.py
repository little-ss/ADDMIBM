#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd


def read_table_auto(path):
    """
    Automatically detect delimiter and load a table file.
    Supports txt / csv / tsv.
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {path}\n{e}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_first_existing_column(df, candidates, file_path, file_role):
    """
    Find the first existing column name from a list of candidates.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise ValueError(
        f"Cannot find a valid {file_role} column in: {file_path}\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected one of: {candidates}"
    )


def normalize_filename_series(series):
    """
    Normalize path strings for matching.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
    )


def map_label_value(x):
    """
    Map various label strings to binary labels.
    1 -> bonafide
    0 -> spoof
    """
    s = str(x).strip().lower()

    bonafide_set = {
        "bonafide", "bona-fide", "bona_fide", "real", "genuine", "human", "1"
    }
    spoof_set = {
        "spoof", "fake", "forged", "synthetic", "tts", "vc", "0"
    }

    if s in bonafide_set:
        return 1
    if s in spoof_set:
        return 0

    return None


def infer_label_from_filename(path_str):
    """
    Infer label from filename/path when no explicit label column is available.
    This is only used as a fallback.
    """
    s = str(path_str).strip().lower().replace("\\", "/")

    bonafide_keywords = [
        "/bonafide/",
        "/bona-fide/",
        "/real/",
        "/genuine/",
    ]

    spoof_keywords = [
        "/spoof/",
        "/fake/",
        "/generated/",
        "_generated/",
        "_gen.",
        "_gen.mp3",
        "/tts/",
        "/vc/",
        "/world_generated/",
    ]

    for kw in bonafide_keywords:
        if kw in s:
            return 1

    for kw in spoof_keywords:
        if kw in s:
            return 0

    return None


def deduplicate_by_column(df, col_name):
    """
    Deduplicate rows by a specific column and keep the first occurrence.
    No warning message will be printed.
    """
    dup_mask = df[col_name].duplicated(keep="first")
    if dup_mask.any():
        df = df.loc[~dup_mask].copy()
    return df


# ---------------------------------------------------------------------
# Official-style EER implementation
# ---------------------------------------------------------------------
def compute_det_curve(target_scores, nontarget_scores):
    """
    Compute DET curve values.

    target_scores: bonafide scores
    nontarget_scores: spoof scores

    Returns:
      frr, far, thresholds
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((
        np.atleast_1d(1),
        nontarget_trial_sums / nontarget_scores.size
    ))
    thresholds = np.concatenate((
        np.atleast_1d(all_scores[indices[0]] - 0.001),
        all_scores[indices]
    ))

    return frr, far, thresholds


def compute_eer_official(target_scores, nontarget_scores):
    """
    Official-style EER computation.

    Returns:
      eer, frr, far, thresholds, eer_threshold
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, frr, far, thresholds, thresholds[min_index]


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
def prepare_score_df(score_file):
    """
    Load and standardize the score file.
    """
    df = read_table_auto(score_file)

    filename_col = find_first_existing_column(
        df,
        ["filename", "file", "path", "wav_path", "audio", "utt_id", "id"],
        score_file,
        "score filename"
    )
    score_col = find_first_existing_column(
        df,
        ["cm-score", "cm_score", "score", "scores", "logit"],
        score_file,
        "score value"
    )

    out = df[[filename_col, score_col]].copy()
    out.columns = ["filename", "cm-score"]
    out["filename"] = normalize_filename_series(out["filename"])
    out["cm-score"] = out["cm-score"].astype(float)

    out = deduplicate_by_column(out, "filename")
    return out


def prepare_label_df(label_file):
    """
    Load and standardize the label/protocol file.
    Supports explicit label columns, and also supports fallback label inference from filename.
    """
    df = read_table_auto(label_file)

    filename_col = find_first_existing_column(
        df,
        ["filename", "file", "path", "wav_path", "audio", "utt_id", "id"],
        label_file,
        "label filename"
    )

    out = df.copy()
    out["filename_std"] = normalize_filename_series(out[filename_col])

    label_col = None
    for cand in ["cm_label", "label", "target", "class", "key"]:
        for c in out.columns:
            if c.lower() == cand.lower():
                label_col = c
                break
        if label_col is not None:
            break

    if label_col is not None:
        out["label"] = out[label_col].apply(map_label_value)
    else:
        out["label"] = out["filename_std"].apply(infer_label_from_filename)

    if out["label"].isnull().any():
        bad_rows = out[out["label"].isnull()][["filename_std"]].head(10)
        raise ValueError(
            f"Failed to determine labels from protocol file: {label_file}\n"
            f"Please make sure the file contains a valid label column such as cm_label/label.\n"
            f"Example unresolved rows:\n{bad_rows.to_string(index=False)}"
        )

    out["label"] = out["label"].astype(int)

    label_out = out[["filename_std", "label"]].copy()
    label_out.columns = ["filename", "label"]

    label_out = deduplicate_by_column(label_out, "filename")
    return label_out


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_one_language(score_file, label_file, save_merged=""):
    """
    Evaluate one language pair: score file + label file.
    Only exact filename matching is used.
    """
    score_df = prepare_score_df(score_file)
    label_df = prepare_label_df(label_file)

    merged = pd.merge(
        score_df[["filename", "cm-score"]],
        label_df[["filename", "label"]],
        on="filename",
        how="inner"
    )

    merged = deduplicate_by_column(merged, "filename")

    if len(merged) == 0:
        raise ValueError(
            f"No matched utterances were found between:\n"
            f"  score_file = {score_file}\n"
            f"  label_file = {label_file}"
        )

    merged["label"] = merged["label"].astype(int)
    merged["cm-score"] = merged["cm-score"].astype(float)

    bonafide_num = int((merged["label"] == 1).sum())
    spoof_num = int((merged["label"] == 0).sum())

    if bonafide_num == 0 or spoof_num == 0:
        raise ValueError(
            f"After merging, one class is empty.\n"
            f"bonafide={bonafide_num}, spoof={spoof_num}\n"
            f"score_file={score_file}\n"
            f"label_file={label_file}"
        )

    # Align score direction with official convention:
    # higher score -> more likely bonafide
    bona_mean = merged.loc[merged["label"] == 1, "cm-score"].mean()
    spoof_mean = merged.loc[merged["label"] == 0, "cm-score"].mean()

    scores = merged["cm-score"].values.copy()
    if bona_mean < spoof_mean:
        scores = -scores
        score_direction = "reversed automatically"
    else:
        score_direction = "original"

    bonafide_scores = scores[merged["label"].values == 1]
    spoof_scores = scores[merged["label"].values == 0]

    eer, frr, far, thresholds, threshold = compute_eer_official(
        bonafide_scores,
        spoof_scores
    )

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
        "merge_mode": "exact filename",
    }


def print_result_block(name, result):
    """
    Print one language result block.
    """
    print(f"[{name}]")
    print(f"Threshold           : {result['threshold']:.10f}")
    print(f"EER                 : {result['eer'] * 100:.9f}%")
    print("-" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Compute official-style EER for CVoice CN/EN/DE/FR/IT and report the average EER."
    )

    parser.add_argument("--cn_score_file", type=str, required=True, help="Path to Chinese score file")
    parser.add_argument("--cn_label_file", type=str, required=True, help="Path to Chinese protocol/label file")

    parser.add_argument("--en_score_file", type=str, required=True, help="Path to English score file")
    parser.add_argument("--en_label_file", type=str, required=True, help="Path to English protocol/label file")

    parser.add_argument("--de_score_file", type=str, required=True, help="Path to German score file")
    parser.add_argument("--de_label_file", type=str, required=True, help="Path to German protocol/label file")

    parser.add_argument("--fr_score_file", type=str, required=True, help="Path to French score file")
    parser.add_argument("--fr_label_file", type=str, required=True, help="Path to French protocol/label file")

    parser.add_argument("--it_score_file", type=str, required=True, help="Path to Italian score file")
    parser.add_argument("--it_label_file", type=str, required=True, help="Path to Italian protocol/label file")

    parser.add_argument("--save_cn_merged", type=str, default="", help="Optional path to save merged CN details")
    parser.add_argument("--save_en_merged", type=str, default="", help="Optional path to save merged EN details")
    parser.add_argument("--save_de_merged", type=str, default="", help="Optional path to save merged DE details")
    parser.add_argument("--save_fr_merged", type=str, default="", help="Optional path to save merged FR details")
    parser.add_argument("--save_it_merged", type=str, default="", help="Optional path to save merged IT details")

    args = parser.parse_args()

    results = {}

    results["CN"] = evaluate_one_language(
        args.cn_score_file, args.cn_label_file, args.save_cn_merged
    )
    results["EN"] = evaluate_one_language(
        args.en_score_file, args.en_label_file, args.save_en_merged
    )
    results["DE"] = evaluate_one_language(
        args.de_score_file, args.de_label_file, args.save_de_merged
    )
    results["FR"] = evaluate_one_language(
        args.fr_score_file, args.fr_label_file, args.save_fr_merged
    )
    results["IT"] = evaluate_one_language(
        args.it_score_file, args.it_label_file, args.save_it_merged
    )

    avg_eer = np.mean([results[k]["eer"] for k in ["CN", "EN", "DE", "FR", "IT"]])

    print("=" * 72)
    print("CVoice 5-Language Evaluation Result")
    print("=" * 72)

    for lang in ["CN", "EN", "DE", "FR", "IT"]:
        print_result_block(lang, results[lang])

    print("[AVG]")
    print(f"AVG EER         : {avg_eer * 100:.9f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()