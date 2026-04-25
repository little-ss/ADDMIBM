# ADDMIBM

**ADDMIBM: An Audio Deepfake Detection System with Multi-Information Cross-Aggregation and Bidirectional Information Modeling**

ADDMIBM is an audio deepfake detection system designed to improve robustness and generalization under complex acoustic conditions, unseen spoofing attacks, and multilingual scenarios. The proposed system integrates self-supervised acoustic representation learning, multi-information cross-aggregation, bidirectional temporal modeling, and adaptive bidirectional information fusion.

## Overview

The overall framework of ADDMIBM consists of four main components:

1. **SSL-based front-end feature extractor**  
   Wav2vec 2.0 / XLS-R is used to extract deep acoustic representations from raw speech signals.

2. **Multi-Information Cross-Aggregation (MICA)**  
   MICA enhances spoofing-related discriminative cues by jointly modeling channel-wise, temporal, and frequency-domain information.

3. **Bidirectional Mamba**  
   Bidirectional Mamba captures both forward and backward long-range temporal dependencies and structural anomaly patterns in speech.

4. **Bidirectional Information Fusion (BIF)**  
   BIF adaptively fuses forward and backward information through local-global attention mapping to improve spoofing artifact perception.

The proposed method is evaluated on seven audio deepfake detection datasets:

- ASVspoof 2019 LA
- ASVspoof 2021 LA
- ASVspoof 2021 DF
- ASVspoof 5
- In-the-Wild (ITW)
- DECRO
- CVoiceFake

## Repository Structure

```text
ADDMIBM/
├── scores/
│   ├── 19LA.txt
│   ├── 21LA.txt
│   ├── 21DF.txt
│   ├── ASV5.txt
│   ├── ITW.txt
│   ├── DECRO/
│   │   ├── DECRO_cn.txt
│   │   └── DECRO_en.txt
│   └── CVoice/
│       ├── CVoice_cn.txt
│       ├── CVoice_en.txt
│       ├── CVoice_de.txt
│       ├── CVoice_fr.txt
│       └── CVoice_it.txt
├── protocols/
│   ├── ASVspoof2019.LA.cm.eval.trl.txt
│   ├── ASVspoof2019.LA.asv.eval.gi.trl.scores.txt
│   ├── ASVspoof5.eval.track_1.evalkey.tsv
│   ├── ITW.txt
│   ├── DECRO_cn.tsv
│   ├── DECRO_en.tsv
│   ├── CVioce_cn.tsv
│   ├── CVioce_en.tsv
│   ├── CVioce_de.tsv
│   ├── CVioce_fr.tsv
│   └── CVioce_it.tsv
├── LA-keys-stage-1/
├── DF-keys-stage-1/
├── evaluation-package/
├── evaluation_19LA.py
├── evaluate_2021_LA.py
├── evaluate_2021_DF.py
├── evaluate_CVoice.py
├── evaluate_DECRO.py
├── evaluate_ITW.py
└── README.md
```

> Note: The protocol files for CVoiceFake are kept as `CVioce_*.tsv` in the commands below to remain consistent with the current file names in this repository. If the files are renamed to `CVoice_*.tsv`, please update the corresponding paths accordingly.

## Evaluation Metrics

Different official protocols are used for different datasets.

| Dataset | Metric |
|---|---|
| ASVspoof 2019 LA | EER, min t-DCF |
| ASVspoof 2021 LA | EER, min t-DCF |
| ASVspoof 2021 DF | EER |
| ASVspoof 5 | EER, minDCF |
| In-the-Wild | EER |
| DECRO | EER |
| CVoiceFake | EER |

Lower values indicate better detection performance.

## Evaluation Commands

All commands should be executed from the root directory of this repository.

### ASVspoof 2019 LA

```bash
python evaluation_19LA.py \
  --score_file "./scores/19LA.txt" \
  --protocol_file "./protocols/ASVspoof2019.LA.cm.eval.trl.txt" \
  --asv_score_file "./protocols/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
```

### ASVspoof 2021 LA

```bash
python evaluate_2021_LA.py \
  ./scores/21LA.txt \
  ./LA-keys-stage-1/keys \
  eval
```

### ASVspoof 2021 DF

```bash
python evaluate_2021_DF.py \
  ./scores/21DF.txt \
  ./DF-keys-stage-1/keys \
  eval
```

### ASVspoof 5

```bash
python ./evaluation-package/evaluation.py \
  --m t1 \
  --cm ./scores/ASV5.txt \
  --cm_key ./protocols/ASVspoof5.eval.track_1.evalkey.tsv
```

### CVoiceFake

```bash
python evaluate_CVoice.py \
  --cn_score_file "./scores/CVoice/CVoice_cn.txt" \
  --cn_label_file "./protocols/CVioce_cn.tsv" \
  --en_score_file "./scores/CVoice/CVoice_en.txt" \
  --en_label_file "./protocols/CVioce_en.tsv" \
  --de_score_file "./scores/CVoice/CVoice_de.txt" \
  --de_label_file "./protocols/CVioce_de.tsv" \
  --fr_score_file "./scores/CVoice/CVoice_fr.txt" \
  --fr_label_file "./protocols/CVioce_fr.tsv" \
  --it_score_file "./scores/CVoice/CVoice_it.txt" \
  --it_label_file "./protocols/CVioce_it.tsv"
```

### DECRO

```bash
python evaluate_DECRO.py \
  --cn_score_file "./scores/DECRO/DECRO_cn.txt" \
  --cn_label_file "./protocols/DECRO_cn.tsv" \
  --en_score_file "./scores/DECRO/DECRO_en.txt" \
  --en_label_file "./protocols/DECRO_en.tsv"
```

### In-the-Wild

```bash
python evaluate_ITW.py \
  --score_file "./scores/ITW.txt" \
  --label_file "./protocols/ITW.txt"
```

## Main Results

The following results are obtained with the default ADDMIBM configuration.

| Dataset | EER (%) | Additional Metric |
|---|---:|---:|
| ASVspoof 2019 LA | 0.11 | min t-DCF: 0.0037 |
| ASVspoof 2021 LA | 0.72 | min t-DCF: 0.2043 |
| ASVspoof 2021 DF | 1.82 | - |
| ASVspoof 5 | 5.56 | minDCF: 0.158 |
| In-the-Wild | 5.96 | - |
| DECRO | 0.05 | - |
| CVoiceFake | 0.49 | - |

## Notes

- Please make sure that all score files and protocol/key files are placed in the correct directories before running the evaluation scripts.
- The utterance IDs in each score file must be consistent with the corresponding protocol or key file.
- For ASVspoof 2021 LA and ASVspoof 2021 DF, the official key directories are required.
- For ASVspoof 5, the official evaluation package is required.
- For multilingual datasets such as DECRO and CVoiceFake, each language subset is evaluated separately and summarized by the evaluation script.
- If path errors occur, please check whether the directory names and file names are consistent with the commands above.

## Citation

If you find this repository useful, please cite our work:

```bibtex
@article{su2026addmibm,
  title={ADDMIBM: An Audio Deepfake Detection System with Multi-Information Cross-Aggregation and Bidirectional Information Modeling},
  author={Su, Wenjie and Huang, Zhihua and Sang, Jialei and Li, Bowen and Wu, Bixin},
  journal={Preprint},
  year={2026}
}
```
