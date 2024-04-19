"""
Text Summarization
- Text summarization is the process of distilling the most important information from a source text.
"""
from tqdm import tqdm

import torch
import torch.nn.functional as F

from pytorch_lightning import seed_everything

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def custom_greedy(model, tok_context, **gen_config):
	"""
	Custom greedy decoding
	:param model:
	:param tok_context:
	:param gen_config:
	:return:
	"""
	MAX_GEN_TOKENS = gen_config.pop("max_new_tokens", 100)
	CHUNK_SIZE = model.config.max_position_embeddings
	TEMP = gen_config.pop("temperature", 1.0)
	DO_SAMPLE = gen_config.pop("do_sample", False)
	
	context = tok_context['input_ids']
	past_key_values = None
	
	model.eval()
	
	with torch.no_grad():
		# Generate new tokens - Sequential Decoding
		for _ in tqdm(range(MAX_GEN_TOKENS)):
			block_context = context[:, -CHUNK_SIZE:]
			model_out = model(block_context, past_key_values)
			logits = model_out.logits / TEMP
			probs = F.softmax(logits[:, -1, :], dim=-1)
			
			# if DO_SAMPLE:
			# 	new_token = torch.multinomial(probs, 1)
			# else:
			
			new_token = torch.argmax(probs, dim=-1, keepdim=True)
			context = torch.cat([context, new_token], dim=-1)
	
	return context


def custom_beam_search(model, tok_context, **gen_config):
	"""
	Custom beam search decoding
	:param model:
	:param tf_inputs:
	:param gen_config:
	:return:
	"""
	MAX_GEN_TOKENS = gen_config.pop("max_new_tokens", 100)
	CHUNK_SIZE = model.config.max_position_embeddings
	TEMP = gen_config.pop("temperature", 1.0)
	
	pass


def decode(
		model,
		tokenizer,
		context,
		max_output_len,
		decoding_strategy: str = 'greedy'
):
	"""
	Decoder output
	:param model:
	:param tokenizer:
	:param context:
	:param max_output_len:
	:param decoding_strategy:
	:return:
	"""
	# Tokenize the context
	tokenized_context = tokenizer(context, return_tensors='pt')
	
	# Generate the output
	if decoding_strategy == 'greedy':
		output = model.generate(
			**tokenized_context,
			max_new_tokens=max_output_len,
			early_stopping=True,
			do_sample=False
		)
		custom_output = custom_greedy(
			model=model,
			tok_context=tokenized_context,
			max_new_tokens=max_output_len,
			temperature=1.0,
			do_sample=False
		)
	
	else:
		raise NotImplementedError
	
	# Decode the output
	decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
	custom_decoded_output = tokenizer.decode(custom_output[0], skip_special_tokens=True)
	
	print("{} Output:\n" + 100 * '-'.format(decoding_strategy))
	print(decoded_output)
	print("{} Custom Implementation Output:\n" + 100 * '-'.format(decoding_strategy))
	print(custom_decoded_output)
	
	return decoded_output


if __name__ == '__main__':
	# Example taken from: https://huggingface.co/blog/how-to-generate
	seed = 42
	nw = 'distilbert/distilgpt2'
	max_gen_seq_len = 100
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	seed_everything(seed=seed)
	LLM = AutoModelForCausalLM.from_pretrained(nw).to(device)
	tok = AutoTokenizer.from_pretrained(nw)
	print("Number of parameters:", LLM.num_parameters())
	
	context_sent = 'I enjoy walking with my cute dog'
	
	decoder_output = decode(
		model=LLM,
		tokenizer=tok,
		context=context_sent,
		max_output_len=max_gen_seq_len
	)


