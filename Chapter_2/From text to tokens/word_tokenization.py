# the simplest class of word tokenization is using whitespace
text = "Tokenizing text is a core task of NLP."
tokenized_text = text.split()
print(tokenized_text)

"""
Having a large vocabulary is a problem because it requires neural networks to have an enormous number of parameters. To illustrate this, suppose we have 1 million unique words and want to compress the 1-million-dimensional input vectors to 1-thousand- dimensional vectors in the first layer of our neural network. This is a standard step in most NLP architectures, and the resulting weight matrix of this first layer would contain 1 million × 1 thousand = 1 billion weights. This is already comparable to the largest GPT-2 model,4 which has around 1.5 billion parameters in total!

A compromise between character and word tokenization is subword tokenization.
"""
