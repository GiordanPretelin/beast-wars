"""
The main distinguishing feature of subword tokenization (as well as word tokenization) is that it is learned from the pre‐training corpus using a mix of statistical rules and algorithms.

There are several subword tokenization algorithms that are commonly used in NLP, but let's start with WordPiece,5 which is used by the BERT and DistilBERT tokenizers.
"""

from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

"""
The AutoTokenizer class belongs to a larger set of “auto” classes whose job is to automatically retrieve the model's configuration, pretrained weights, or vocabulary from the name of the checkpoint. This allows you to quickly switch between models, but if you wish to load the specific class manually you can do so as well. For example, we could have loaded the DistilBERT tokenizer as follows:
from transformers import DistilBertTokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
"""

text = "Tokenizing text is a core task of NLP."
encoded_text = tokenizer(text)
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

print(tokenizer.convert_tokens_to_string(tokens))
print(f"tokenizer.vocab_size: {tokenizer.vocab_size}")
print(f"tokenizer.model_max_length: {tokenizer.model_max_length}")
print(f"tokenizer.model_input_names: {tokenizer.model_input_names}")
