
""" This code is adapted from huggingface examples for finetuning transformers on sequence classification tasks
    and desired changes are made to train it on our task."""

import argparse
import logging
import math
import os
import random
import glob
import matplotlib.pyplot as plt

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN
import transformers
from accelerate import Accelerator
from models import BertForMultiTaskClassification
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from utils import plot_confusion_matrix

ACT2FN["leaky_relu"] = F.leaky_relu
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=["EmotionDetector", "NextEmotionPredictor"], 
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="If passed, will start training.",
    )
    parser.add_argument(
        "--va",
        type=int,
        default=0,
        help="Number of dimentions for reggresion. (2 for VA and 3 for VAD)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--eval_on_test",
        action="store_true",
        help="If passed, will use evaluate on test data.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="If passed, will use evaluate all checkpoints on test data.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Loading the dataset from local csv or json file.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
    raw_datasets = load_dataset("csv", data_files=data_files, delimiter="\t")
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = raw_datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    print(f"Number of Labels = {num_labels}, \n {label_list}")
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = BertConfig.from_pretrained(args.model_name_or_path, hidden_act="leaky_relu", hidden_dropout_prob=0.5, num_labels=num_labels, finetuning_task="EmotionDetector", id2label={str(i): label for i, label in enumerate(label_list)}, label2id={label: i for i, label in enumerate(label_list)})
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForMultiTaskClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        va = args.va
    )

    sentence1_key, sentence2_key = "Utterance", None

    label_to_id = {v: i for i, v in enumerate(label_list)}
    print(f"\nlabels to id: {label_to_id}")
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        
        if "vad_label" in examples and args.va != 0:
            result["va_labels"] = [eval(e) for e in examples["vad_label"]]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    test_dataset = processed_datasets["test"]
    eval_dataset = processed_datasets["validation"]
    print(f"Length of data sets = {len(train_dataset)}, {len(test_dataset)}, {len(eval_dataset)}")
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )


    metric = load_metric("/data/Kunal/cheerbots/accuracy")

    if args.do_train:
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        train_losses = []
        eval_losses = []
        va_train_lossess = []
        va_eval_lossess = []
        for epoch in range(args.num_train_epochs):
            model.train()
            train_loss = 0.0
            va_train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.cls_loss
                train_loss += loss.item()
                if outputs.va_loss is not None:
                    loss += outputs.va_loss
                    va_train_loss += outputs.va_loss.item()

                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break

            avg_train_loss = train_loss / len(train_dataloader)
            avg_va_train_loss = va_train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)
            va_train_lossess.append(avg_va_train_loss)

            model.eval()
            eval_loss = 0.0
            va_eval_loss = 0.0
            for step, batch in enumerate(eval_dataloader):
                outputs = model(**batch)
                eval_loss += outputs.cls_loss.item()
                if outputs.va_loss is not None:
                    va_eval_loss += outputs.va_loss.item()

                predictions = outputs.cls_logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

            avg_eval_loss = eval_loss / len(eval_dataloader)
            avg_va_eval_loss = va_eval_loss / len(eval_dataloader)
            eval_losses.append(avg_eval_loss)
            va_eval_lossess.append(avg_va_eval_loss)

            eval_metric = metric.compute()
            cf_matrix = eval_metric["cf_matrix"]
            norm_cf_matrix = eval_metric["norm_cf_matrix"]
            eval_metric["avg_train_loss"] = avg_train_loss
            eval_metric["avg_va_train_loss"] = avg_va_train_loss
            eval_metric["avg_eval_loss"] = avg_eval_loss
            eval_metric["avg_va_eval_loss"] = avg_va_eval_loss
            del eval_metric["cf_matrix"]
            del eval_metric["norm_cf_matrix"]
            logger.info(f"epoch {epoch}: {eval_metric}")

            if args.output_dir is not None:
                with open(os.path.join(args.output_dir, f"results.txt"), 'a') as out:
                    out.write(f"\nepoch {epoch}: {str(eval_metric)}")

                accelerator.wait_for_everyone()
                output_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plot_confusion_matrix(cf_matrix, norm_cf_matrix, label_list, output_dir, 'eval', epoch)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info(f"Saving model checkpoint to {output_dir}")
        
        plt.plot(range(len(train_losses)), train_losses, label='Train')
        plt.plot(range(len(eval_losses)), eval_losses, label='Valid')
        if args.va != 0:
            plt.plot(range(len(va_train_lossess)), va_train_lossess, label='va_Train')
            plt.plot(range(len(va_eval_lossess)), va_eval_lossess, label='va_Valid')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, "loss_curve.svg"), format='svg', dpi=600)
        plt.close()

    if args.test_file is not None and args.eval_on_test:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            # checkpoints = checkpoints[-1:]
            checkpoints = ["/data/Kunal/cheerbots/emotion_detector_with_vad_all_utterances-1.8.0/checkpoint-8"]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        
        logger.info("Evaluating the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            epoch = checkpoint.split("-")[-1]
            checkpoint1 = os.path.join(checkpoint, "converted_to_old_version")
            # state_dict = torch.load(os.path.join(checkpoint, "pytorch_model.bin"), map_location="cpu")
            test_model = BertForMultiTaskClassification.from_pretrained(checkpoint1, config=checkpoint, va=args.va)
            # torch.save(test_model.state_dict(), os.path.join(checkpoint1, "pytorch_model.bin"), _use_new_zipfile_serialization=False)
            test_model, test_dataloader = accelerator.prepare(test_model, test_dataloader)
            test_model.eval()
            test_loss = 0.0
            va_test_loss = 0.0
            for step, batch in enumerate(test_dataloader):
                outputs = test_model(**batch)
                test_loss += outputs.cls_loss.item()
                if outputs.va_loss is not None:
                    va_test_loss += outputs.va_loss.item()
                    
                predictions = outputs.cls_logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )
            avg_test_loss = test_loss / len(test_dataloader)
            avg_va_test_loss = va_test_loss / len(test_dataloader)

            test_metric = metric.compute()
            cf_matrix = test_metric["cf_matrix"]
            norm_cf_matrix = test_metric["norm_cf_matrix"]
            test_metric["avg_test_loss"] = avg_test_loss
            test_metric["avg_va_test_loss"] = avg_va_test_loss
            del test_metric["cf_matrix"]
            del test_metric["norm_cf_matrix"]
            plot_confusion_matrix(cf_matrix, norm_cf_matrix, label_list, checkpoint, 'train_on_utter_test_on_situations', epoch)
            logger.info(f"epoch {epoch}: {test_metric}")
        
            with open(os.path.join(args.output_dir, f"train_on_utter_test_on_situations_results.txt"), 'a') as out:
                out.write(f"\nepoch {epoch}: {str(test_metric)}")

if __name__ == "__main__":
    main()
