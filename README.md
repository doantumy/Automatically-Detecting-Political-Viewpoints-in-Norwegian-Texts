# Automatically Detecting Political Viewpoint in Norwegian Texts


# Downstream Tasks Experiments

## Training instructions
- Trainings were done on NTNU IDUN cluster using NVIDIA A100 40GB GPU
- Replace the value of `--model` with one of the following:
  - `north/t5_base_NCC`: north T5
  - `ltg/nort5-base`: norT5 by University of Oslo
  - `google/mt5-base`: mT5 by Google
  - `./sp-t5-base`: SP-T5-base (our model)
  - `./sp-t5-base-kw`: SP-T5-base-kw (our model)
- Change `--output_dir` folder to corresponding model name to make sure checkpoints are saved properly.
- For logging train/eval/test into `wandb` tracker, please use `--use_wandb`:
  - Change `--wandb_prefix` to corresponding model name
  - and provide `--wandb_project_name`. For example: `summary`, `translation`, etc.


## Task 1: Political Viewpoint Identification (PVI)
> **Evaluation metric**
   ROUGE1/2/3/L and human evaluation

```python
# This works for T5 model
python3 encoder_decoder_sum.py \
  --model=ltg/nort5-base \
  --train_file="./viewpoint_train.tsv" \
  --validation_file="./viewpoint_validation.tsv" \
  --test_file="./viewpoint_test.tsv" \
  --output_dir="nort5_vp" \
  --task_prefix="finn synspunkt: " \
  --max_length="512" \
  --epochs="10" \
  --batch_size="16" --lr="5e-5" \
  --max_new_token="100" --min_length="50" \
  --do_train --do_eval --do_predict \
  --use_wandb --wandb_prefix=nort5_vp --wandb_project_name="viewpoint"
# our SP-models are trained on TPU using Flax framework, need to add --use_flax into the training command
```

## Task 2: Political Speech Summarization (Norwegian only)
> **Evaluation metric**
   ROUGE1/2/3/L

```python
# This works for T5 model
python3 encoder_decoder_sum.py \
  --model=ltg/nort5-base \
  --train_file="./summaries_train.tsv" \
  --validation_file="./summaries_validation.tsv" \
  --test_file="./summaries_test.tsv" \
  --output_dir="nort5_sum" \
  --task_prefix="oppsummer: " \
  --max_length="512" \
  --epochs="10" \
  --batch_size="16" --lr="4e-5" \
  --do_train --do_eval --do_predict \
  --no_repeat_ngram_size="3" --num_beams="8" --max_new_token="250" --min_length="100" \
  --use_wandb --wandb_prefix="nort5_sum" --wandb_project_name="summary"
# our SP-models are trained on TPU using Flax framework, need to add --use_flax into the training command
```

## Task 3: Translation EU Parliament Speeches
> **Evaluation metric**
   `BLEU` score

### `Danish` to `Swedish`
```python
# This works for T5 model
python3 encoder_decoder_trans.py \
  --model=./sp-t5-base \
  --source_lang="da" \
  --target_lang="sv" \
  --train_file="./europarl/europarl_train.tsv" \
  --validation_file="./europarl/europarl_val.tsv" \
  --test_file="./europarl/europarl_test.tsv" \
  --output_dir="spt5_trans" \
  --task_prefix="oversæt fra dansk til svensk: " \
  --max_length="512" \
  --epochs="5" \
  --batch_size="32" \
  --do_predict --do_train --do_eval \
  --max_new_token="256" --min_length="24" --lr="5e-5" \
  --no_repeat_ngram_size="2" --num_beams="2" \
  --use_wandb --wandb_prefix="spt5_trans" \
  --wandb_project_name="translation_da_sv" --use_flax
# our SP-models are trained on TPU using Flax framework, need to add --use_flax into the training command
```
### `Swedish` to `Danish`
- For translating from Swedish to Danish, change the `--source_lang` to `sv` and `--target_lang` to `da`
- Update the `--task_prefix` to `"översätt från svenska till danska: `
- [Optional] Change the name of the `--output_dir`, `--wandb_prefix` and `--wandb_project_name`

## Task 4: Political Leaning (Left/Right) For Norwegian and Swedish Parliament Speeches

> **Evaluation metric**
  Accuracy and $F_1$ score

### A. For Norwegian
- `Left/Right` is `venstre/høyre` in Norwegian. This is set by default in `--label_list_per_language`.
- Task prefix `find political leaning: ` is translated to Norwegian `finn politisk tilhørighet: `

```python
# This works for T5 model
python3 encoder_decoder_cls.py \
  --model=./sp-t5-base \
  --task_name="leaning_classification" \
  --train_file="./nor-train.tsv" \
  --validation_file="./no-val.tsv" \
  --test_file="./no-test.tsv" \
  --output_dir="spt5_leaning_no" \
  --task_prefix="finn politisk tilhørighet: " \
  --max_length="512" \
  --epochs="10" \
  --batch_size="32" \
  --do_predict --do_train --do_eval \
  --max_new_token="2" --min_length=1 --lr="5e-5" \
  --use_wandb --wandb_prefix=spt5_leaning_no --wandb_project_name="no_leaning" --use_flax
# our SP-models are trained on TPU using Flax framework, need to add --use_flax into the training command
```

### B. For Swedish
- Need to update `--label_list_per_language="vänster,höger"`
- Task prefix `find political leaning: ` is translated to Swedish `finn politisk tilhørighet: `
```python
# This works for T5 model
python3 encoder_decoder_cls.py \
  --model=./sp-t5-base \
  --task_name="leaning_classification" \
  --train_file="./se-train.tsv" \
  --validation_file="./se-val.tsv" \
  --test_file="./se-test.tsv" \
  --output_dir=spt5_leaning_se \
  --task_prefix="hitta politisk tillhörighet: " \
  --max_length="512" \
  --epochs="10" \
  --batch_size="32" \
  --do_predict --do_train --do_eval --label_list_per_language="vänster,höger" \
  --max_new_token="2" --min_length="1" --lr="5e-5" \
  --use_wandb --wandb_prefix=spt5_leaning_se --wandb_project_name="leaning_se" \
  --no_repeat_ngram_size=2 --num_beams=2 --use_flax
# our SP-models are trained on TPU using Flax framework, need to add --use_flax into the training command
```

# Training T5 without keyword masking
> **NOTES:** Models will be uploaded on Huggingface later.

- Training on TPU v3-8, batch size of `128`, sequence length of `512`
- Model was continued to train on `north/t5_base_NCC`.
- `north/t5_base_NCC` was trained on NCC dataset and some additional data from MC4 and English Wikipedia for `500k` steps from `mT5` checkpoint on TPU v4-8.
- Training file [🤗 training-file](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py#L827)
- Training command:

```python
python3 run_t5_mlm_flax.py \
    --output_dir="sp-t5-base" \
    --model_type="t5" \
    --overwrite_output_dir=True \
    --model_name_or_path="north/t5_base_NCC" \
    --train_file="./data/noswdaisTrainingSorted20230911.txt" \
    --validation_file="./data/noswdaisValidationSorted20230911.txt" \
    --do_train=True \
    --do_eval=True \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --max_seq_length="512" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --adafactor \
    --learning_rate="3e-5" \
    --weight_decay="0.1" \
    --warmup_steps="18000" \
    --seed="123" \
    --num_train_epochs="150" \
    --logging_steps="1000" \
    --eval_steps="10000" \
    --save_steps="10000" \
    --cache=".cache" \
    --wandb_project_name="sp-t5-base" \
    --wandb_entity="llm" \
    --preprocessing_num_workers="64" 
```


# Training T5 model with keyword masking
- Training on TPU v2-8, total batch size of `96`, equence length of `512`
- Number of opinion keywords: `1,822`
## Prepare masking file
```python3
# mkdir t5-mask
# python3 apply_t5_tokenizer.py
# This will output a list of tokens, write to mask.json file
{
  "to_mask": [
    [
      7025
    ],
    [
      66,
      6420,
      18
    ],
    ...],
  "mode": "keyword" 
}
```

## Training command

- Training on TPU v2-8, total batch size of `96`
- Number of opinion keywords: `1,822`

```python
python3 run_t5_mlm_flax_kw.py \
    --to_mask_file=./t5-mask/mask.json \
    --output_dir="sp-t5-base-kw" \
    --model_type="t5" \
    --overwrite_output_dir=True \
    --model_name_or_path="./sp-t5-base" \
    --train_file="./data/noswdaisTrainingSorted20230911.txt" \
    --validation_file="./data/noswdaisValidationSorted20230911.txt" \
    --do_train=True \
    --do_eval=True \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --max_seq_length="512" \
    --per_device_train_batch_size="12" \
    --per_device_eval_batch_size="12" \
    --adafactor \
    --learning_rate="3e-5" \
    --weight_decay="0.1" \
    --warmup_steps="6250" \
    --seed="123" \
    --num_train_epochs="100" \
    --logging_steps="1000" \
    --eval_steps="10000" \
    --save_steps="10000" \
    --cache=".cache" \
    --wandb_project_name="sp-t5-base-kw" \
    --wandb_entity="llm" \
    --preprocessing_num_workers="64"
```
## Link to models on 🤗 Huggingface

[🤗 spt5-keyword](https://huggingface.co/tumd/spt5-kw/)

[🤗 spt5](https://huggingface.co/tumd/spt5/)
