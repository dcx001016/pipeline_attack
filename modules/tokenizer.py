from transformers import GPT2TokenizerFast, DebertaV2Tokenizer

def build_tokenizer(args):
    # tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)
    if args.tokenizer_name == "gpt2":
        tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer/gpt2-GPT2TokenizerFast")
    elif args.tokenizer_name == "gpt2-xl":
        tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer/gpt2-xl-GPT2TokenizerFast")
    elif args.tokenizer_name == "gpt2-medium":
        tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer/gpt2-medium-GPT2TokenizerFast")
    elif args.tokenizer_name == "gpt2-large":
        tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer/gpt2-large-GPT2TokenizerFast")
    else: 
        tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def build_deberta_tokenizer(args):
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.tokenizer_name)
    return tokenizer