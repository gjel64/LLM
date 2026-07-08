import tiktoken

def createTokenizer(special_tokens):
    tokenizer = tiktoken.get_encoding("p50k_base")
    base = len(tokenizer._mergeable_ranks)

    existing = set(tokenizer._special_tokens.keys())

    new_tokens = {
        token: base + i
        for i, token in enumerate(special_tokens)
        if token not in existing  # évite les conflits
    }

    extended_tokenizer = tiktoken.Encoding(
        name="p50k_custom",
        pat_str=tokenizer._pat_str,
        mergeable_ranks=tokenizer._mergeable_ranks,
        special_tokens={**tokenizer._special_tokens, **new_tokens},
    )
    return extended_tokenizer

# 
# tokenizer = createTokenizer(["<|startoftext|>", "<|endoftext|>"])
# print(tokenizer.encode(
#     "<|startoftext|> hello <|endoftext|>",
#     allowed_special="all"
# ))