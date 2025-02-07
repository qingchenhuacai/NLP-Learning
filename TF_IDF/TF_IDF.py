import jieba
import math
import os
from collections import defaultdict
import json

class TFIDF:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.corpus = [] # 存放分词后的语料库，每个元素是list，是每篇文章的分词结果
        self.tf_dict = defaultdict(lambda: defaultdict(int)) # key:文档序号，value：dict，文档中每个词出现的频率
        self.idf_dict = defaultdict(set) # key:词， value：set，词出现过的文档序号，最终用于计算每个词在多少篇文档中出现过
        self.tf_idf_dict = defaultdict(lambda: defaultdict(float)) # key:文档序号，value：dict，文档中每个词的tf-idf值
        # self.load_corpus_from_txt(self.corpus_path)
        # self.build_tf_idf()

    def load_corpus_from_txt(self, corpus_path):
        # 读取语料库文件夹，将txt文件读取成str存入corpus []
        for path in os.listdir(corpus_path):
            path = os.path.join(corpus_path, path)
            if path.endswith('txt'):
                with open(path, 'r', encoding='utf-8') as f:
                    self.corpus.append(f.read()) # f.read()返回str
        
        self.corpus = [jieba.lcut(text) for text in self.corpus] # 返回嵌套列表，每个text被分词后组成list
        return
    
    def load_corpus_from_json(self, corpus_path):
        '''
        [
            {
                "title": "叠穿：越多穿越瘦是可能的（组图）",
                "content": "编者按：今年冬季让叠穿打造你的完美身材，越穿越瘦是可能。哪怕是不了解流行，也要对时尚又显瘦的叠穿造型吸引，现在就开始行动吧！搭配Tips：亮红色的皮外套给人光彩夺目的感觉，内搭短版的黑色T恤，露出带有线条的腹部是关键，展现你健美的身材。 搭配Tips：简单款型的机车装也是百搭的单品，内搭一条长版的连衣裙打造瘦身的中性装扮。软硬结合的mix风同样备受关注。 搭配Tips：贴身的黑色装最能达到瘦身的效果，即时加上白色的长外套也不会发福。长款的靴子同样很好的修饰了你的小腿线条。 搭配Tips：高腰线的抹胸装很有拉长下身比例的效果，A字形的荷叶摆同时也能掩盖腰部的赘肉。外加一件短款的羽绒服，配上贴腿的仔裤，也很修长。"
            }
        ]
        '''
        # 读取语料库文件夹，将json文件读取成str存入corpus []
        for path in os.listdir(corpus_path):
            path = os.path.join(corpus_path, path)
            print(path)
            if path.endswith('json'):
                with open(path, 'r', encoding='utf-8') as f:
                    documents = json.loads(f.read())
                    for document in documents:
                        self.corpus.append(document['title'] + '/n' + document['content'])
        return
    
    def build_tf_idf(self):
        for text_id, text in enumerate(self.corpus):
            for word in text:
                self.tf_dict[text_id][word] += 1 # 每个word在每篇文章text_id中出现的次数
                self.idf_dict[word].add(text_id) # 每个word出现过的文档序号
        # 把idf_dict中每个词的值转为文档数
        for word, text_id_set in self.idf_dict.items():
            self.idf_dict[word] = len(text_id_set) 
        
        # 计算tf-idf
        total_texts = len(self.corpus) # 总文档数
        for text_id, word_count_dict in self.tf_dict.items():
            total_words = sum(word_count_dict.values()) # 某篇文档的总词数
            for word, count in word_count_dict.items():
                tf = count / total_words                
                idf = math.log(total_texts / (self.idf_dict[word] + 1)) 
                tf_idf = tf * idf
                self.tf_idf_dict[text_id][word] = tf_idf
        return
    
    def get_top_k(self, k, if_print = True):
        topk_dict = {}
        for text_id, word_tf_idf_dict in self.tf_idf_dict.items():
            sorted_tdidf_list = sorted(word_tf_idf_dict.items(), key=lambda x: x[1], reverse=True)[:k]
            topk_dict[text_id] = sorted_tdidf_list
            if if_print:
                print(f"第{text_id}篇文章的top {k}个词：")
                for i in range(k):
                    print(sorted_tdidf_list[i])
                print("----------")
        return topk_dict


if __name__ == '__main__':
    
    txt_corpus_path = "TF_IDF/corpus/category_corpus/"
    tfidf = TFIDF(txt_corpus_path)
    tfidf.load_corpus_from_txt(txt_corpus_path)
    tfidf.build_tf_idf()
    topk_dict = tfidf.get_top_k(10, if_print=True)
    
    json_corpus_path = "TF_IDF/corpus/"
    tfidf = TFIDF(json_corpus_path)
    tfidf.load_corpus_from_json(json_corpus_path)
    tfidf.build_tf_idf()
    topk_dict = tfidf.get_top_k(10, if_print=True)


