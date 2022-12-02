from datasets import load_dataset, DatasetDict
import os
import re
import csv
import random
import sys

version = "large_text_only"    # 2m_text_only (some bug exists) or large_text_only
output_dir = "../dataset/dataset_part/"
train_file = output_dir + "train.csv"
validation_file = output_dir + "validation.csv"
test_file = output_dir + "test.csv"
end = -1  # -1 means process all

train_portion = 0.08
validation_portion = 0.01
test_portion = 0.01


def is_description(clause):
    return len(clause.split()) >= 5


def construct_data(description, tag):
    merge_tag = random.randint(0, 3)
    input = description + ', ' + ', '.join(tag[0:merge_tag])
    output = ', '.join(tag[merge_tag:])
    return input, output


def process_data(output_file, dataset):
    i = 0
    file = open(output_file, 'w')
    writer = csv.writer(file)
    writer.writerow(['original_prompt', 'description', 'tag', 'input', 'output'])

    for prompt in dataset['prompt']:
        description = []
        tag = []
        strings = re.split(r'[,.;]', prompt)

        for clause in strings:
            if clause == '':
                continue
            if clause[0] == ' ':
                clause = clause[1:]
            if clause == '':
                continue
            if is_description(clause):
                description.append(clause)
            else:
                tag.append(clause)

        if len(description) == 0 or len(tag) < 6:
            continue

        description = ', '.join(description)

        input, output = construct_data(description, tag)

        writer.writerow([prompt, description, tag, input, output])
        i += 1

        if i % 10000 == 0:
            print(str(i)+": done")

        if end != -1 and i > end:
            break


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = load_dataset("poloclub/diffusiondb", version)
    print(dataset)

    # split data set
    train_testvalidr = dataset['train'].train_test_split(test_size=1-train_portion)
    # Split the 10% test + valid in half test, half valid
    red_testvalid = train_testvalidr['test'].train_test_split(test_size=test_portion + validation_portion)
    test_valid = red_testvalid['test'].train_test_split(test_size=test_portion / (test_portion + validation_portion))
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_ds = DatasetDict({
        'train': train_testvalidr['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']})
    print(train_test_valid_ds)

    process_data(test_file, train_test_valid_ds['test'])
    process_data(validation_file, train_test_valid_ds['validation'])
    process_data(train_file, train_test_valid_ds['train'])


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == "sample":
            output_dir = "../dataset/dataset_sample/"
            end = 1000  # -1 means process all
            train_file = output_dir + "train.csv"
            validation_file = output_dir + "validation.csv"
            test_file = output_dir + "test.csv"
    main()
