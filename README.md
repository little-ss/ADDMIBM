19LA

python evaluation_19LA.py --score_file "./scores/19LA.txt" \
  --protocol_file "./protocols/ASVspoof2019.LA.cm.eval.trl.txt" \
  --asv_score_file "./protocols/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"

21LA

python evaluate_2021_LA.py ./scores/21LA.txt ./LA-keys-stage-1/keys eval

21DF

python evaluate_2021_DF.py ./scores/21DF.txt ./DF-keys-stage-1/keys eval

ASV5

python ./evaluation-package/evaluation.py --m t1 --cm ./scores/ASV5.txt --cm_key ./protocols/ASVspoof5.eval.track_1.evalkey.tsv

CVoice

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

DECRO

python evaluate_DECRO.py \
  --cn_score_file "./scores/DECRO/DECRO_cn.txt" \
  --cn_label_file "./protocols/DECRO_cn.tsv" \
  --en_score_file "./scores/DECRO/DECRO_en.txt" \
  --en_label_file "./protocols/DECRO_en.tsv"

ITW

python evaluate_ITW.py   --score_file "./scores/ITW.txt"   --label_file "./protocols/ITW.txt"
