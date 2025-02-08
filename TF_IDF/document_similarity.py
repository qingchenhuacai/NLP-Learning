import jieba
import TF_IDF
import math
from collections import defaultdict
import numpy as np

'''
基于语料库中所有文章的TF-IDF值，收集每篇文章的top_k关键词（使用get_top_k）
遍历每篇文章，统计每个关键词的词频，记录为每篇文章的向量
输入查询语句，同样生成查询语句的词频向量，计算与每篇文章的余弦相似度
输出相似度最高的几篇文章
'''

def generate_key_words_list(topk_dict:dict) -> list:
    # {文档id : [(词1，从高到低的tf_idf值)，(词2，从高到低的tf_idf值)，(词3，从高到低的tf_idf值)]}
    key_word_set = set()
    for topk_words_with_tfidf in topk_dict.values():
        for word, tf_idf in topk_words_with_tfidf:
            key_word_set.add(word)
    return list(key_word_set)

def calc_vector(document:str, key_word_list:list) -> list[float]:
    # 输入一篇文档和关键词list
    # 给文档分词，依次使用 .count() 查询每个关键词的词频，除文档总词数
    words = jieba.lcut(document)
    vector = []
    for key_word in key_word_list:
        vector.append( words.count(key_word) / len(words) )
    return vector

def generate_corpus_vector(corpus:list, key_word_list:list) -> dict[list]:
    # 为语料库的每篇文章生成一个关键词频向量，记录为dict
    corpus_vector_dict = defaultdict()
    for index, document in enumerate(corpus):
        corpus_vector_dict[index] = calc_vector(document, key_word_list)
    return corpus_vector_dict

def cosine_similarity(vec1:list, vec2:list) -> float:
    # 计算余弦相似度
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

def find_most_similar_doc(input:str, key_word_list:list, corpus_vector_dict:dict, corpus:list, topk=3) -> list[str]:
    vec_input = calc_vector(input, key_word_list)
    res = []
    highest_index = []
    for index, doc_vec in corpus_vector_dict.items():
        similarity = cosine_similarity(vec_input, doc_vec)
        highest_index.append((index, similarity))
    
    highest_index = sorted(highest_index, key = lambda x:x[1], reverse = True)[:topk]
    for index, similarity in highest_index:
        res.append(corpus[index])
    return res

if __name__ == '__main__':
    json_corpus_path = "TF_IDF/corpus/"
    tf_idf = TF_IDF.TFIDF(json_corpus_path)
    tf_idf.load_corpus_from_json(json_corpus_path)
    tf_idf.build_tf_idf()
    corpus = tf_idf.corpus
    key_word_list = generate_key_words_list(tf_idf.get_top_k(3, if_print = False))
    corpus_vector_dict = generate_corpus_vector(corpus, key_word_list)
    input = '魔兽世界世界首杀诞生'
    res = find_most_similar_doc(input, key_word_list, corpus_vector_dict, corpus)
    for i in res:
        print(i + '\n')