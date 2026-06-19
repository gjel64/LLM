import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
i = 10000
while(True):
    print(f"{i} : {enc.decode([i])}")
    i += 1