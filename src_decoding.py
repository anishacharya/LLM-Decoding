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
	
	context = tok_context['input_ids']
	past_key_values = None
	
	model.eval()
	
	with torch.no_grad():
		# Generate new tokens - Sequential Decoding
		for _ in tqdm(range(MAX_GEN_TOKENS)):
			block_context = context[:, -CHUNK_SIZE:]
			model_out = model(block_context, past_key_values)
			logits = model_out.logits  # / TEMP
			probs = F.softmax(logits[:, -1, :], dim=-1)
			
			# if DO_SAMPLE:
			# 	new_token = torch.multinomial(probs, 1)
			# else:
			
			new_token = torch.argmax(probs, dim=-1, keepdim=True)
			context = torch.cat([context, new_token], dim=-1)
	
	return context


def decode(
		model,
		tokenizer,
		context,
		max_output_len,
		decoding_strategy: str = 'greedy'
):
	"""
	Perform Decoding for Generation Tasks
	
	:param model: Pre-trained language model
	:param tokenizer: Tokenizer
	:param context:  Context
	:param max_output_len: Maximum output length
	:param decoding_strategy: Decoding strategy
	
	:return: Decoded output
	"""
	# Tokenize the context
	tok_context = tok(context, return_tensors='pt')
	tok_context = {k: v.to(device) for k, v in tok_context.items()}
	
	# Generate the output
	if decoding_strategy == 'greedy':
		output = model.generate(
			**tok_context,
			max_new_tokens=max_output_len,
			early_stopping=True,
			do_sample=False
		)
		custom_output = custom_greedy(
			model=model,
			tok_context=tok_context,
			max_new_tokens=max_output_len,
			early_stopping=True,
			do_sample=False
		)
		assert (output[i] == custom_output[i] for i in range(len(output)))
	
	else:
		raise NotImplementedError
	
	# Decode the generated output
	decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
	print("{} Decoding Output:\n" + 100 * '-'.format(decoding_strategy))
	print(decoded_output)
	
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
	
	test_x = 'I enjoy walking with my cute dog'
	
	decoder_output = decode(
		model=LLM,
		tokenizer=tok,
		context=test_x,
		max_output_len=max_gen_seq_len
	)


