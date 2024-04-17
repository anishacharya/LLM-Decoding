"""
Text Summarization
"""
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import numpy as np
import random


if __name__ == '__main__':
	nw = 'distilbert/distilgpt2'
	max_gen_seq_len = 10
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	model = AutoModelForCausalLM.from_pretrained(nw).to(device)
	tokenizer = AutoTokenizer.from_pretrained(nw)
	
	context = 'I enjoy walking with my cute dog'
	tokenized_context = tokenizer(context, return_tensors='pt')
	
	greedy_output = model.generate(
		**tokenized_context,
		max_new_tokens=max_gen_seq_len,
		do_sample=False
	)
	
	print("Output:\n" + 100 * '-')
	print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))