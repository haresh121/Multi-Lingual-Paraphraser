from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders
import os

print(os.getcwd())
tokenizer = Tokenizer.from_file("../tokenizers/paraBert.json")
# bert_tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
# bert_tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
# bert_tokenizer.pre_tokenizer = Whitespace()
# bert_tokenizer.post_processor = TemplateProcessing(
#     single="<s> $A </s>",
#     pair="<s> $A <s> $B </s>",
#     special_tokens=[
#         ("<s>", 0),
#         ("</s>", 1)
#     ]
# )
#
# trainer = WordPieceTrainer(vocab_size=10_00_000, special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"])
# files = ["../data/clean-data/data.csv", "../data/words/words.txt"]
#
# bert_tokenizer.train(files, trainer)
#
# bert_tokenizer.save("../tokenizers/paraBert.json", pretty=True)

tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>", length=50)
output = tokenizer.encode("Hello this is using a library. this is very cool to work with!")
# tokenizer.decoder = decoders.WordPiece()
print(output.ids)
print(tokenizer.decode(output.ids))
