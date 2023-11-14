from tokenizers import Tokenizer
from tokenizers.models import BPE
from datasets import load_dataset
import pyarrow as pa
from tokenizers.pre_tokenizers import Metaspace, Whitespace
from tokenizers.trainers import BpeTrainer



def train_tokenizer(vocab_size=None, texts= None):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # tokenizer.pre_tokenizer = Metaspace() # Painfully slow
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
    if texts is None:
        dataset = load_dataset("pg19")
        texts = [text for text in dataset['train']['text']]

    tokenizer.train_from_iterator(texts, trainer) 

    if vocab_size is not None:
        tokenizer.save(f"bpe-tokenizer-{vocab_size}.json")
    else: 
        tokenizer.save("bpe-tokenizer.json")

if __name__ == "__main__":
    dataset = load_dataset("pg19")
    texts = [text for text in dataset['train']['text']]
    
    train_tokenizer(5000, texts)
    tokenizer = Tokenizer.from_file("bpe-tokenizer-5000.json")
    # tokenizer.pre_tokenizer = Metaspace()
    tokenizer.pre_tokenizer = Whitespace()
    ex = texts[0][:1000]
    print(ex)
    print(tokenizer.encode(ex).tokens)
