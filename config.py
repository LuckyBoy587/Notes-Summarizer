import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import threading

# Lazy-loaded model/tokenizer to avoid expensive import-time work
_MODEL = None
_TOKENIZER = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
_LOCK = threading.Lock()

def get_model_tokenizer_device(use_fp16_on_cuda=True):
	"""Return (model, tokenizer, device) and load them on first call. Thread-safe."""
	global _MODEL, _TOKENIZER, _DEVICE
	
	# Fast path: if already loaded, return immediately
	if _MODEL is not None and _TOKENIZER is not None:
		return _MODEL, _TOKENIZER, _DEVICE
	
	# Slow path: load with lock
	with _LOCK:
		# Double-check after acquiring lock
		if _TOKENIZER is None:
			_TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
		if _MODEL is None:
			if use_fp16_on_cuda and torch.cuda.is_available():
				_MODEL = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME, torch_dtype=torch.float16)
			else:
				_MODEL = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
			_MODEL = _MODEL.to(_DEVICE)
			_MODEL.eval()
	return _MODEL, _TOKENIZER, _DEVICE

def get_device():
	return _DEVICE