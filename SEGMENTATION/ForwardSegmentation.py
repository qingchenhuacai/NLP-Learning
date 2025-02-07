import torch
import torch.nn as nn
from collections import defaultdict
from itertools import islice

class ForwardSegmentation:
    def __init__(self, dict_path: str):
        self.word_dict = defaultdict()
        self.max_len = 0
        self.load_dict(dict_path)
        self.word_prefix_dict = defaultdict()
        self.load_prefix_dict(dict_path)

    def load_dict(self, dict_path: str) -> tuple[dict, int]:
        '''
        读取词表，返回词表字典和最长词长度
        词表示例：
        一万次 16 m
        一万步 30 m
        一万盏 4 m
        '''
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.split()[0]
                self.word_dict[word] = len(word)
                self.max_len = max(self.max_len, len(word))
    
    def segmentation(self, text: str) -> list[str]:
        '''
        利用词表进行正向最大分词
        '''
        word_seg = []
        while text:
            word = text[:self.max_len]
            while word not in self.word_dict:
                if len(word) == 1:
                    break
                word = word[:-1]
            word_seg.append(word)
            text = text[len(word):]
        return word_seg

    def load_prefix_dict(self, dict_path: str) -> dict:
        '''
        读取词表，返回词前缀字典
        '''
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.split()[0]
                for i in range(len(word)):
                    if word[:i+1] not in self.word_prefix_dict: # 避免把词记成前缀
                        self.word_prefix_dict[word[:i+1]] = 0
                self.word_prefix_dict[word] = 1

    def segmentation_prefix(self,text: str) -> list[str]:
        '''
        用前缀此表进行分词
        '''
        word_seg = []
        start_index, end_index = 0,1
        window = text[start_index:end_index]
        if_word = window # 第一个字作为初始的潜在词
        while start_index < len(text):
            if window not in self.word_prefix_dict or end_index > len(text):
                # 单字不在词表中，直接记录
                # 窗口后移，新窗口不在前缀词表中，则if_word是最长的词
                # 或者窗口已经到达文本末尾，则直接记录if_word
                word_seg.append(if_word)
                start_index += len(if_word)
                end_index = start_index + 1
                window = text[start_index:end_index] # 新的单字窗口
                if_word = window
            elif self.word_prefix_dict[window] == 0: # 找到了前缀，继续扩大窗口
                end_index += 1
                window = text[start_index:end_index]
            elif self.word_prefix_dict[window] == 1: # 找到了词，记录为潜在的if_word，窗口后移检查会不会有更长的词
                if_word = window
                end_index += 1
                window = text[start_index:end_index]
        return word_seg
            


if __name__ =='__main__':
    text_path = './input/corpus.txt'
    dict_path = './input/dict.txt'
    output_path = './output/corpus_res1.txt'
    writer = open(output_path, 'w', encoding='utf-8')
    seg = ForwardSegmentation(dict_path)
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = seg.segmentation(line.strip())
            writer.write('/'.join(words) + '\n')
    writer.close()
    print('forward segmentation done')
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in islice(f, 10):
            print(line.strip())
    
    sample = "王羲之草书《平安帖》共有九行"
    print(seg.segmentation_prefix(sample))