"""
Text Summarization
- Text summarization is the process of distilling the most important information from a source text.

Generate HF
https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L1224
"""
from tqdm import tqdm
from typing import Dict

import torch
import torch.nn.functional as F

from pytorch_lightning import seed_everything

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def custom_greedy(
		model,
		tok_context,
		max_new_tokens: int = 40,
):
	"""
	Custom Greedy decoding :
	
	:param model: Pre-trained language model
	
	:param tok_context: Tokenized context (input_ids)
	
	:param max_new_tokens: Maximum number of tokens to generate
	
	:return:
	"""
	MAX_GEN_TOKENS = max_new_tokens
	CHUNK_SIZE = model.config.max_position_embeddings
	
	curr_gen = tok_context['input_ids']  # initialized with the input_ids
	past_key_values = None
	
	model.eval()
	
	with torch.no_grad():
		# Generate new tokens - Sequential Decoding
		for _ in tqdm(range(MAX_GEN_TOKENS)):
			block_context = curr_gen[:, -CHUNK_SIZE:]
			model_out = model(block_context, past_key_values)
			logits = model_out.logits  # / TEMP
			logits = logits[:, -1, :]  # Last token Logits
			probs = F.softmax(logits, dim=-1)
			
			# if DO_SAMPLE:
			# 	new_token = torch.multinomial(probs, 1)
			# else:
			
			new_token = torch.argmax(probs, dim=-1, keepdim=True)
			curr_gen = torch.cat([curr_gen, new_token], dim=-1)
	
	return curr_gen


def decode(
		model,
		tokenizer,
		context,
		max_output_len: int = 100,
		num_beams: int = 5,
		decoding_strategy: str = 'greedy'
):
	"""
	Perform Decoding for Generation Tasks
	
	:param model: Pre-trained language model
	:param tokenizer: Tokenizer
	:param context:  Context
	:param max_output_len: Maximum output length
	:param num_beams: Number of beams for beam search
	
	:param decoding_strategy: Decoding strategy
	
	:return: Decoded output
	"""
	# Tokenize the context
	tok_context = tok(context, return_tensors='pt')
	tok_context = {k: v.to(device) for k, v in tok_context.items()}
	
	model.eval()
	
	with torch.no_grad():
		# Generate the output
		if decoding_strategy == 'greedy':
			output = model.generate(
				**tok_context,
				max_new_tokens=max_output_len,
			)
			custom_output = custom_greedy(
				model=model,
				tok_context=tok_context,
				max_new_tokens=max_output_len,
			)
			assert (output[i] == custom_output[i] for i in range(len(output)))
		
		elif decoding_strategy == 'beam_search':
			# activate beam search and early_stopping
			output = model.generate(
				**tok_context,
				max_new_tokens=max_output_len,
				num_beams=num_beams
			)
		
		else:
			raise NotImplementedError
	
	# Decode the generated output
	decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
	
	return decoded_output


if __name__ == '__main__':
	# Example taken from:
	# https://huggingface.co/blog/how-to-generate
	seed = 42
	nw = 'gpt2'
	max_gen_seq_len = 100
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	seed_everything(seed=seed)
	
	tok = AutoTokenizer.from_pretrained(nw)
	LLM = AutoModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path=nw,
		pad_token_id=tok.eos_token_id
	)
	LLM = LLM.to(device)
	print("Number of parameters:", LLM.num_parameters())
	
	test_x = 'I enjoy walking with my cute dog'
	
	# greedy_output = decode(
	# 	model=LLM,
	# 	tokenizer=tok,
	# 	context=test_x,
	# 	max_output_len=max_gen_seq_len,
	# 	decoding_strategy='greedy'
	# )
	# print(f"Greedy Decoding\n" + 100 * "-")
	# print(greedy_output)
	
	BEAM_SIZE = 2
	
	beam_search_output = decode(
		model=LLM,
		tokenizer=tok,
		context=test_x,
		max_output_len=max_gen_seq_len,
		num_beams=BEAM_SIZE,
		decoding_strategy='beam_search'
	)
	print(f"Beam Search Decoding\n" + 100 * "-")
	print(beam_search_output)


