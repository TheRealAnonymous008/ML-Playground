TODOs
1. Improve training -- training should be done so that lower resource languages train with more epochs (i.e., they train epoch / max(epoch))

2. Semantics and Syntax matching -- the way to do this is as follows: First generate the entire vocabulary. Then, establish word frequency via a Zipf distribution (may be optional). Then, tokenize the vocabulary to match with the language model (i.e., a pretrained GPT2 / LLAMA instance). Finally, when generating, output the tokens and decode using our new token. 

Training Scheme for MoE
Each GPT corresponds to 1 language

Training is done as follows for each language :
    - Run the forward loop as normal. This means that the other GPTs will interfere with this language in a sense because their outputs are not masked out.
    - Update the loss of only the language mode