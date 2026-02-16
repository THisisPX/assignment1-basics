
# from collections import 
import regex as re
class BPE_Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None)
        PAT = r"""'(?:[sdmt][ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\n{L}\p{N}]+|\s+(?|\S)|\s+"""

        ## 构造函数，接收以下参数创建分词器：
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        # for i in range(256):
        #     self.vocab ={bytes([i]):i}
        #     self.vocab_v = {i:bytes([i])}
        self.vocab = vocab
        for i in range(256):
            self.vocab ={bytes([i]):i}
            self.vocab_v = {i:bytes([i])}
        self.merges = merges
        self.special_tokens =special_tokens


    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)

        ## 类方法，从序列化的词汇表文件和合并记录文件（格式应与BPE训练代码输出一致）构造并返回Tokenizer实例，
        ## 接收参数：
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None

        with open(vocab_filepath, "r") as f:
            self.vocab = f
        with open(merges_filepath, "r") as f:
            self.merges = f


    def encode(self, text: str) -> list[int]
        ## 将输入文本编码为token ID序列
        sections = re.split("|".join(self.special_tokens), text)
        for section in sections :
            words = re.finditer(PAT,section)
            for 

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
        ## 接收字符串可迭代对象（如Python文件句柄），返回惰性生成token ID的生成器。
        ## 该方法用于高效处理无法直接加载到内存的大文件

    def decode(self, ids: list[int]) -> str
        ##将token ID序列解码为文本

    def train(self, text, )











    def __init__():
        self.vocab ={}
        self.vocab_v = {}
        for i in range(256):
            self.vocab ={bytes([i]):i}
            self.vocab_v = {i:bytes([i])}

        

