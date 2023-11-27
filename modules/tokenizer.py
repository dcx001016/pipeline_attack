from transformers import AutoTokenizer

def build_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer