from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import evaluate
from transformers import Seq2SeqTrainingArguments
import numpy as np
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
import nltk
import sys
nltk.download('punkt')

model_output_dir = "../models/t5_large_data/"
data_path = "../dataset/dataset_large/"
# begin_checkpoint = "facebook/bart-base"
begin_checkpoint = "t5-base"
train_epochs = 8
max_input_length = 256
max_target_length = 512


def main():
    # load and split data set
    ds = load_dataset("csv", data_files=data_path+"train.csv")
    ds['test'] = load_dataset("csv", data_files=data_path+"test.csv")['train']
    ds['validation'] = load_dataset("csv", data_files=data_path+"validation.csv")['train']
    print(ds)

    for i in range(0, train_epochs):
        if i == 0:
            model_checkpoint = begin_checkpoint
        else:
            model_checkpoint = model_output_dir + "epoch" + str(i - 1)
        output_dir_epoch = model_output_dir + "epoch" + str(i)
        train_one_epoch(model_checkpoint, output_dir_epoch, i, ds)


def training_args(data_num):
    batch_size = 8
    num_train_epochs = 1  # one at a time and save models
    # Show the training loss with every epoch
    logging_steps = data_num // batch_size
    model_name = "prompt_helper"

    args = Seq2SeqTrainingArguments(
        output_dir=model_name,
        evaluation_strategy="epoch",
        learning_rate=5.6e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
    )

    return args


def train_one_epoch(model_checkpoint, output_dir, epoch, dataset):

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # load models
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # preprocess dataset
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            examples["output"], max_length=max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels_mask"] = labels["attention_mask"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    print(tokenized_datasets)

    # evaluation metrics
    rouge_score = evaluate.load("rouge")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    # initialize trainer
    args = training_args(len(tokenized_datasets["train"]))
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    with open(model_output_dir + "eval_epoch_"+str(epoch)+".txt", "w") as file1:
        file1.write(str(trainer.evaluate()))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == "sample":
            model_output_dir = "../models/t5_sample_data/"
            data_path = "../dataset/dataset_sample/"
            train_epochs = 2
    main()
