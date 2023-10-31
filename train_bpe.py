from tokenizers import Tokenizer
from tokenizers.models import BPE
from datasets import load_dataset
import pyarrow as pa
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer



def train_tokenizer(vocab_size=None, texts= None):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Metaspace()
    # tokenizer.pre_tokenizer = Whitespace()

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
    # arrow_files = ["pg19/default-2b2ff260da3cc3e2/0.0.0/74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137/pg19-train-00000-of-00015.arrow"]

    # # Accumulate text data from all files
    # all_texts = []

    # for file_path in arrow_files:
    # # Load Arrow file
    #     table = pa.read_table(file_path)

    #     # Convert to Pandas DataFrame
    #     df = table.to_pandas()

    #     # Extract the text data
    #     texts = df['text_column'].tolist()  # Replace 'text_column' with the appropriate column name

    #     all_texts.extend(texts)


    
    train_tokenizer(500, texts[:100])
    tokenizer = Tokenizer.from_file("bpe-tokenizer-500.json")
    tokenizer.pre_tokenizer = Metaspace()
    ex = texts[0][:1000]
    print(ex)
    print(tokenizer.encode(ex).tokens)