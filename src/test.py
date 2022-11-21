# from transformers import BartTokenizer, AutoModelForSeq2SeqLM
#
# output_dir = "my_models2/epoch2"
# model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
# tokenizer = BartTokenizer.from_pretrained(output_dir)
#
# raw_inputs = ["god, holy"]
# inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
#
# outputs = model.generate(inputs["input_ids"])
#
# print(raw_inputs)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
