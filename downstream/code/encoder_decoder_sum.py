import argparse
import torch
import random
import math
import nltk
import os
import numpy as np
from collections import defaultdict
from statistics import mean, stdev
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset import Dataset, EncoderDecoderCollator
# from datasets import load_metric
import evaluate
import wandb
import csv
import shutil
import re

def copy_config_files(src_dir, dst_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Iterate through all files in the source directory and copy them
    for filename in os.listdir(src_dir):
        src_file_path = os.path.join(src_dir, filename)
        dst_file_path = os.path.join(dst_dir, filename)
        
        # Check if it's a file
        if os.path.isfile(src_file_path):
            shutil.copy2(src_file_path, dst_file_path)
                
def write_batch_results(preds, labels, file_name):
    # Check if file exists
    if not os.path.exists(file_name):
        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["preds", "labels"])
    # append data to file
    with open(file_name, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")    
        for p, l in zip(preds, labels):
            writer.writerow([p,l])

def postprocess_preds(preds):
    preds = [re.sub(r"<extra_id_\d+>", "", pred.lower().strip()).strip() for pred in preds]
    return preds

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
        
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="google/mt5-base", type=str)
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--validation_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--output_dir", default="summary", type=str)
    parser.add_argument("--task_prefix", default="oppsummere: ", type=str)
    parser.add_argument("--lr", default=2.0e-5, type=float, help="learning rate.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="learning rate.")
    parser.add_argument("--warmup_portion", default=0.06, type=float, help="learning rate.")
    parser.add_argument("--max_length", default=512, type=int, help="max_length for data collator")
    parser.add_argument("--acummulation_steps", default=1, type=int)
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument('--mixed_precision', action="store_true", default=False)
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--do_predict", action="store_true", default=False)
    parser.add_argument("--temperature", default=.2, type=float)
    parser.add_argument("--length_penalty", default=None, type=float)
    parser.add_argument("--num_beams", default=None, type=int)
    parser.add_argument("--top_k", default=None, type=int)
    parser.add_argument("--top_p", default=None, type=float)
    parser.add_argument("--no_repeat_ngram_size", default=2, type=int)
    parser.add_argument("--wandb_prefix", default="gpt2_sum", type=str)
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Log values using wandb")
    parser.add_argument("--wandb_project_name", default=None, type=str)
    parser.add_argument("--use_flax", action="store_true", default=False, help="Load flax model instead of pytorch")
    parser.add_argument("--max_new_token", default=50, type=int)
    parser.add_argument("--min_length", default=30, type=int)
    parser.add_argument("--load_best_at_the_end", action="store_true", default=False)

    args = parser.parse_args()

    return args

def setup_training(seed, args):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    return device

if __name__ == "__main__":
    args = parse_arguments()
    
    seed_results = []
    train_set = Dataset(args.train_file)
    valid_set = Dataset(args.validation_file)
    test_set = Dataset(args.test_file)
    
    # Metric
    metric = evaluate.load("rouge")
    
    max_new_token = args.max_new_token #50 #math.floor(args.max_length/7)
    min_length = args.min_length #30 #math.floor(args.max_length/10)
    
    # seed_list = [67, 78, 89, 90, 123]
    seed_list = [123, 456, 789] 
    # seed_list = [123] 
    for seed in seed_list:
        if args.use_wandb: 
            wandb.init(project=args.wandb_project_name, name=f"{args.wandb_prefix}_seed_{seed}", config=args)  # `name` is optional but can be useful to differentiate runs
        device = setup_training(seed, args)
        
        output_dir = f"{args.output_dir}_seed_{seed}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Folder: {} is created.".format(output_dir))
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        if args.use_flax:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model, trust_remote_code=True, from_flax=True).to(device)
        else: model = AutoModelForSeq2SeqLM.from_pretrained(args.model, trust_remote_code=True).to(device)

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size // args.acummulation_steps,
            shuffle=True,
            drop_last=True,
            collate_fn=EncoderDecoderCollator(tokenizer, args.max_length, args.task_prefix),
            num_workers=1,
            pin_memory=True
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size // args.acummulation_steps,
            shuffle=False,
            drop_last=False,
            collate_fn=EncoderDecoderCollator(tokenizer, args.max_length, args.task_prefix),
            num_workers=1,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size // args.acummulation_steps,
            shuffle=False,
            drop_last=False,
            collate_fn=EncoderDecoderCollator(tokenizer, args.max_length, args.task_prefix),
            num_workers=1,
            pin_memory=True
        )

        no_decay = ['bias', "layer_norm", "embedding", "LayerNorm", "Embedding"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
                "lr": args.lr
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": args.lr
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6)

        def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))

                return lr

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        scheduler = cosine_schedule_with_warmup(optimizer, args.epochs*len(train_loader) * args.warmup_portion, args.epochs*len(train_loader), 0.1)

        grad_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
        
        if args.do_train:
            total_train_steps = 0
            best_eval_score = {
                "rouge2": -1,
                "epoch": -1,
                "seed":seed,
            }
            
            for epoch in range(args.epochs):
                # Training
                train_loss, eval_loss = 0. , 0.
                tqdm_bar_train = tqdm(train_loader, desc=f"Epoch: {epoch} Training...", position=0)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for i, batch in enumerate(tqdm_bar_train):
                    source_ids, attention_mask, target_ids = (item.to(device) for item in batch)
                    with torch.cuda.amp.autocast(args.mixed_precision):
                        loss = model(
                            input_ids=source_ids,
                            attention_mask=attention_mask,
                            labels=target_ids
                        ).loss
                        if args.use_wandb:
                            wandb.log({"train/loss": loss.item()}, step=total_train_steps)
                            wandb.log({"train/lr": scheduler.get_last_lr()[0]})
                        if i % 10 == 0 and i > 0:
                            print("Epoch: {}  Loss: {}".format(epoch, round(loss.item(), 7)))
                    grad_scaler.scale(loss / args.acummulation_steps).backward()

                    if (i + 1) % args.acummulation_steps == 0:
                        grad_scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=25.0)
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    total_train_steps += 1
                
                if not args.load_best_at_the_end:
                    model_save_path = f"{output_dir}/checkpoint_epoch_{epoch}"
                    model.save_pretrained(model_save_path)
                    tokenizer.save_pretrained(model_save_path)
                
                # Evaluation
                if args.do_eval:
                    model.eval()
                    with torch.no_grad():
                        tqdm_bar_eval = tqdm(valid_loader, desc=f"Epoch: {epoch} Evaluating...", position=1, leave=False)
                        for i, batch in enumerate(tqdm_bar_eval):
                            optimizer.zero_grad(set_to_none=True)
                            source_ids, attention_mask, target_ids = (item.to(device) for item in batch)

                            with torch.cuda.amp.autocast(args.mixed_precision):
                                targets = target_ids.cpu()
                                predictions = model.generate(
                                    input_ids=source_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=max_new_token,
                                    min_length=min_length,
                                    length_penalty = args.length_penalty,
                                    num_beams=args.num_beams,
                                    early_stopping=True,
                                    temperature=args.temperature,
                                    # repetition_penalty=1.5,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    no_repeat_ngram_size=args.no_repeat_ngram_size
                                ).cpu()
                                if isinstance(predictions, tuple):
                                    predictions = predictions[0]
                                # Replace -100s used for padding as we can't decode them
                                preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                                
                                labels = np.where(targets != -100, targets, tokenizer.pad_token_id)
                                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                                tqdm.write(f"Eval pred: {decoded_preds[0]} <|>")
                                tqdm.write(f"Eval src: {decoded_labels[0]} <|>")

                                # Some simple post-processing
                                decoded_preds = postprocess_preds(decoded_preds)
                                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                        result = metric.compute(use_stemmer=True)
                        result = {k: round(v * 100, 4) for k, v in result.items()}
                        print('result["rouge2"]', result["rouge2"])
                        # if args.load_best_at_the_end:
                        if result["rouge2"] > best_eval_score["rouge2"]:
                            # If there's a previously saved best checkpoint, remove it.
                            if best_eval_score["epoch"] != -1:
                                checkpoint_dir_to_remove = f"{output_dir}/checkpoint_epoch_{best_eval_score['epoch']}/"
                                if os.path.exists(checkpoint_dir_to_remove):
                                    shutil.rmtree(checkpoint_dir_to_remove)
                                    print(f"Removed old best checkpoint: {checkpoint_dir_to_remove}")
                                else:
                                    print(f"Checkpoint {checkpoint_dir_to_remove} not found.")
                            
                            # Save the current model as the new best model.
                            model_save_path = f"{output_dir}/checkpoint_epoch_{epoch}"
                            model.save_pretrained(model_save_path)
                            tokenizer.save_pretrained(model_save_path)
                            print(f"Saving checkpoint: checkpoint_epoch_{epoch}")
                            
                            # Update the best score and epoch
                            best_eval_score["rouge2"] = result["rouge2"]
                            best_eval_score["epoch"] = epoch
                            print("Updated best_eval_score:", best_eval_score)
 
                        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
                        # result["gen_len"] = np.mean(prediction_lens)
                        print("Eval: ", result)
                        if args.use_wandb: 
                            wandb.log({"eval/" + k: v for k,v in result.items()}, step=total_train_steps)
                
        # Testing
        if args.do_predict:
            if not args.load_best_at_the_end:
                model.eval()
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(test_loader)):
                        optimizer.zero_grad(set_to_none=True)
                        source_ids, attention_mask, target_ids = (item.to(device) for item in batch)

                        with torch.cuda.amp.autocast(args.mixed_precision):
                            predictions = model.generate(
                                input_ids=source_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_token,
                                min_length=min_length,
                                length_penalty = args.length_penalty,
                                num_beams=args.num_beams,
                                early_stopping=True,
                                temperature=args.temperature,
                                # repetition_penalty=1.5,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                no_repeat_ngram_size=args.no_repeat_ngram_size
                            ).cpu()
                            targets = target_ids.cpu()

                            if isinstance(predictions, tuple):
                                predictions = predictions[0]
                            # Replace -100s used for padding as we can't decode them
                            preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                            
                            labels = np.where(targets != -100, targets, tokenizer.pad_token_id)
                            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                            tqdm.write(f"Test pred: {decoded_preds[0]} <|>")
                            tqdm.write(f"Test src: {decoded_labels[0]} <|>")

                            # Some simple post-processing
                            decoded_preds = postprocess_preds(decoded_preds)
                            write_batch_results(preds=decoded_preds, labels=decoded_labels, file_name=f"{output_dir}/{output_dir}_epoch_{epoch}.tsv")
                            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                print(f"Writing result file: {output_dir}/{output_dir}_epoch_{epoch}.tsv")
            else:
                # Load best model from evaluation
                best_model_path = f"./{output_dir}/checkpoint_epoch_{best_eval_score['epoch']}/"
                # best_epoch = best_model_path.split("/")[-1].split("_")[-1]
                print(f"Loading best model: {best_model_path} for testing.")
                
                if args.model.split("/")[-1] == "nort5-base":
                    # Need to copy config files to checkpoint before loading
                    # files = ["modeling_nort5.py", "configuration_nort5.py"]
                    copy_config_files("custom_nort5_config" ,best_model_path)
                
                if args.use_flax:
                    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, trust_remote_code=True, from_flax=True).to(device)
                else: model = AutoModelForSeq2SeqLM.from_pretrained(args.model, trust_remote_code=True).to(device)
                
                # model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path, trust_remote_code=True).to(device)
                model.eval()
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(test_loader)):
                        optimizer.zero_grad(set_to_none=True)
                        source_ids, attention_mask, target_ids = (item.to(device) for item in batch)

                        with torch.cuda.amp.autocast(args.mixed_precision):
                            predictions = model.generate(
                                input_ids=source_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_token,
                                min_length=min_length,
                                length_penalty = args.length_penalty,
                                num_beams=args.num_beams,
                                early_stopping=True,
                                temperature=args.temperature,
                                # repetition_penalty=1.5,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                no_repeat_ngram_size=args.no_repeat_ngram_size
                            ).cpu()
                            targets = target_ids.cpu()

                            if isinstance(predictions, tuple):
                                predictions = predictions[0]
                            # Replace -100s used for padding as we can't decode them
                            preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                            
                            labels = np.where(targets != -100, targets, tokenizer.pad_token_id)
                            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                            tqdm.write(f"Test pred: {decoded_preds[0]} <|>")
                            tqdm.write(f"Test src: {decoded_labels[0]} <|>")

                            # Some simple post-processing
                            decoded_preds = postprocess_preds(decoded_preds)
                            write_batch_results(preds=decoded_preds, labels=decoded_labels, file_name=f"{output_dir}/{output_dir}_epoch_{best_eval_score['epoch']}.tsv")
                            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                    print(f"Writing result file: {output_dir}/{output_dir}_epoch_{best_eval_score['epoch']}.tsv")
                    
            result = metric.compute(use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            seed_results.append(result)
            tqdm.write(f"Test ROUGE score: {result}")
            if args.use_wandb:
                wandb.log({"test/" + k: v for k,v in result.items()})
                wandb.finish()
    # Calculate mean
    mean_results = {
        metric: sum([run[metric] for run in seed_results]) / len(seed_list) for metric in seed_results[0]
    }
    mean_results = {k: round(v, 4) for k, v in mean_results.items()}
    # Calculate standard deviation
    std_dev_results = {
        metric: 
        (sum([(run[metric] - mean_results[metric]) ** 2 for run in seed_results]) / len(seed_list)) ** 0.5 
        for metric in seed_results[0]
    }
    std_dev_results = {k: round(v, 4) for k, v in std_dev_results.items()}
    print("Mean Results:", mean_results)
    print("Standard Deviation Results:", std_dev_results)
    
    with open(f"results_{args.output_dir}_{args.model.split('/')[-1]}.txt", 'a') as f:
        for run_time, seed_result in enumerate(seed_results):
            f.write(f"Run: {run_time}\tRouge: {seed_result}\n")
        f.write(f"Mean:\t\tRouge {mean_results}\n")
        f.write(f"Stadard Deviation:\t{std_dev_results}\n")
    print(f"Results have been saved in results_{args.output_dir}_{args.model.split('/')[-1]}.txt.")
   
