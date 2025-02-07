import math
from collections import defaultdict

class NewWordDetect:
    def __init__(self, corpus_path: str):
        self._max_word_len = 4 # 最大词长
        self.corpus_path = corpus_path
        self.word_count = defaultdict(int) # 词频表，记录每个词的出现次数
        # {'北京':3}

        self.l_neighbor = defaultdict(lambda: defaultdict(int)) # 左邻居表，记录每个词的左邻居及其出现次数
        # {'北京': {'a':1, 'b':4}}
        self.r_neighbor = defaultdict(lambda: defaultdict(int)) # 右邻居表，记录每个词的右邻居及其出现次数
        # {'北京': {'A':1, 'B':4}}

        self.word_l_entropy = defaultdict(float) # 左熵表，记录每个词的左熵
        self.word_r_entropy = defaultdict(float) # 右熵表，记录每个词的右熵
        # {'北京': entropy}
        
        self.word_pmi = defaultdict(float) # 词-pmi值表，记录每个词的PMI值
        self.word_value = defaultdict(float) # 词值表，记录每个词的值

        self.load_corpus(corpus_path) # 加载语料库,搭建词频表、左邻居表、右邻居表
        self.build_pmi() # 建立词-pmi值表
        self.build_entropy() # 建立左邻居-熵表，右邻居-熵表
        self.build_word_value() # 计算PMI * min(左熵，右熵)

    def load_corpus(self, corpus_path: str):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip()
                for word_lenth in range(1, self._max_word_len + 1):
                    self.ngram_count(sentence, word_lenth)
        return
    
    def ngram_count(self, sentence: str, word_lenth: int):
        '''
        按词长切割句子，统计词频，统计词的所有左/右邻居及词频
        l_char    word-start    word-inner    word-end    r_char
        i-1           i                   i+word_lenth-1 i+word_lenth
        '''
        # 滑动窗口逐个字符右移取词
        for i in range(len(sentence) - word_lenth + 1):
            # 统计词频
            word = sentence[i:i + word_lenth]
            self.word_count[word] += 1
            # 统计左邻居
            if i > 0:
                l_char = sentence[i-1]
                self.l_neighbor[word][l_char] += 1
            # 统计右邻居
            if i + word_lenth < len(sentence):
                r_char = sentence[i + word_lenth]
                self.r_neighbor[word][r_char] += 1
        return
    
    def calc_entropy(self, word_count_dict: dict) -> float:
        # l_neigbour = {word : word_count_dict}
        # word_count_dict -> entropy
        total_count = sum(word_count_dict.values())
        entropy = 0
        for count in word_count_dict.values():
            probability = count / total_count
            entropy -= probability * math.log(probability,10)
        return entropy
    
    def build_entropy(self):
        # word_l_entropy = {word : l_entropy}
        for word, word_count_dict in self.l_neighbor.items():
            self.word_l_entropy[word] = self.calc_entropy(word_count_dict)
        for word, word_count_dict in self.r_neighbor.items():
            self.word_r_entropy[word] = self.calc_entropy(word_count_dict)
        return
    
    def build_pmi(self):
        self.wordlenth_count = defaultdict(int)
        for word, count in self.word_count.items():
            self.wordlenth_count[len(word)] += count

        for word, count in self.word_count.items():
            p_word = count / self.wordlenth_count[len(word)] # word在等长词中出现的概率
            p_char = 1
            for char in word:
                p_char *= self.word_count[char] / self.wordlenth_count[1]
            
            self.word_pmi[word] = math.log(p_word / p_char, 10) / len(word)
        
        return
    
    def build_word_value(self):
        for word, pmi in self.word_pmi.items():
            if len(word) < 2 or "，" in word:
                continue
            l_entropy = self.word_l_entropy[word]
            r_entropy = self.word_r_entropy[word]
            self.word_value[word] = pmi * min(l_entropy, r_entropy)
        return

if __name__ == "__main__":
    nwd = NewWordDetect("sample_corpus.txt")
    # print(nwd.word_count)
    # print(nwd.left_neighbor)
    # print(nwd.right_neighbor)
    # print(nwd.pmi)
    # print(nwd.word_left_entropy)
    # print(nwd.word_right_entropy)
    value_sort = sorted([(word, count) for word, count in nwd.word_value.items()], key=lambda x:x[1], reverse=True)
    print([x for x, c in value_sort if len(x) == 2][:10])
    print([x for x, c in value_sort if len(x) == 3][:10])
    print([x for x, c in value_sort if len(x) == 4][:10])
            
    
    

