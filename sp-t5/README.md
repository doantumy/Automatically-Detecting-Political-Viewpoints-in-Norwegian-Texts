# Training T5 model with keyword masking
## Prepare masking file
```python3
# mkdir t5-mask
# opinion keywords file: opinion_keywords_for_masking.txt
python3 apply_t5_tokenizer.py
# This will output a list of tokens, saved to mask.json file
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
# set "mode" : "original" to run training without keyword masking
```
## Training command
- Training on TPU v2-8, total batch size of `96`
- Number of opinion keywords: `1,822`

```python
python3 run_t5_mlm_flax_kw.py \
    --to_mask_file=./t5-mask/mask.json \
    --output_dir="kw-sp-t5-base" \
    --model_type="t5" \
    --overwrite_output_dir=True \
    --model_name_or_path="./sp-t5" \
    --train_file="./data/llm_data/noswdaisTraining.txt" \
    --validation_file="./data/llm_data/noswdaisValidation.txt" \
    --do_train=True \
    --do_eval=True \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --max_seq_length="512" \
    --per_device_train_batch_size="12" \
    --per_device_eval_batch_size="12" \
    --adafactor \
    --learning_rate="4e-5" \
    --weight_decay=0.001 \
    --warmup_steps="35000" \
    --seed="123" \
    --num_train_epochs="100" \
    --logging_steps="1000" \
    --eval_steps="10000" \
    --save_steps="10000" \
    --cache=".cache" \
    --wandb_project_name="kw-sp-t5-base" \
    --wandb_entity="llm" \
    --preprocessing_num_workers="128"
```

# Training T5 without keyword masking
- Training on TPU v3-8, batch size of `128`, sequence length of `512`
- Model was continued to train on `north/t5_base_NCC`.
- `north/t5_base_NCC` was trained on NCC dataset and some additional data from MC4 and English Wikipedia for `500k` steps from `mT5` checkpoint on TPU v4-8.
- Training file [🤗 training-file](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py#L827)
- Training command: [🤗 training-command](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling)