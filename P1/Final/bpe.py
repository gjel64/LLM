from tqdm import tqdm
import json

class BPE:
    def __init__(self, text_for_vocab, vocab_size, special_tokens=True, load_from_file=""):

        if (load_from_file == ""):
            self.merges = self._build_merges(text_for_vocab, vocab_size) # (int, int) -> int
        else:
            self._load(load_from_file)
            
        self.vocab = self._build_vocab() # int -> bytes

        # add special tokens for start/end of text
        if (special_tokens):
            special_tokens = {
                len(self.vocab) : b"<|endoftext|>",
                len(self.vocab) + 1 : b"<|startoftext|>",
                len(self.vocab) + 2 : b"<|startthink|>",
                len(self.vocab) + 3 : b"<|endthink|>",
            }
        else :
            special_tokens = {}
        self.vocab.update(special_tokens)

    def _build_merges(self, text_for_vocab, vocab_size):
        """ Create the vocab / merges for the given text and vocab_size """
        bytes_txt = text_for_vocab.encode("utf-8")
        tokens = list(map(int, bytes_txt))
        num_merges = vocab_size - 256
        ids = list(tokens) # copy so we don't destroy the original list

        merges = {} # (int, int) -> int
        for i in tqdm(range(num_merges), desc="building vocab"):
            stats = self._get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self._merge(ids, pair, idx)
            merges[pair] = idx

        return merges

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        return vocab

    def _merge(self, ids, pair, idx):
        """ return new list of ids, with every occurrence of pair replaced by idx """
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def _get_stats(self, ids):
        """ return counts per pair of ids (int, int) : int """
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    

    def decode(self, ids):
        """ ids -> str """
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            # find the highest-priority merge present in tokens
            stats = self._get_stats(tokens)
            # filter to only known merges, pick lowest idx (= earliest merge)
            pair = min(
                (p for p in stats if p in self.merges),
                key=lambda p: self.merges[p],
                default=None
            )
            if pair is None:
                break
            tokens = self._merge(tokens, pair, self.merges[pair])
        return tokens
    
    def save(self, name):
        with open(name, "w") as f:
            f.write(json.dumps({str(k): v for k, v in self.merges.items()}))

    def _load(self, name):
        with open(name, "r") as f:
            merges = json.load(f)
            self.merges = {eval(k): v for k, v in merges.items()}
            

