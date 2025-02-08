import math
import os
import TF_IDF
import re
import jieba

def generate_document_abstracts(document_tf_idf, document, top_k = 3):
    '''
    document_tf_idf: document中每个词的tf-idf值, 保存在TF_IDF.tf_idf_dict : dict中
    document: str, 文档内容，保存在TF_IDF.corpus : list中
    top_k: int, 关键词个数
    '''
    # 切分句子 -> 分词并查询每个词的tf_idf值 -> 按tf_idf值排序 -> 取top_k个词 -> 生成摘要
    # 1. 切分句子
    sentences = re.split("。|！|？", document)
    if len(sentences) < 5:
        return None
    # 2. 分词并查询每个词的tf_idf值
    result = []
    for index, sentence in enumerate(sentences):
        words = jieba.lcut(sentence)
        total_tf_idf = 0
        for word in words:
            total_tf_idf += document_tf_idf.get(word,0)
        result.append([index, total_tf_idf])
    # 3. 按tf_idf值排序
    result.sort(key=lambda x: x[1], reverse=True) # [index, 从高到低的total_tf_idf]
    # 4. 取top_k个词
    result = result[:top_k]
    result.sort(key = lambda x:x[0]) # [前top_k个从底到高的index, total_tf_idf]
    # 5. 生成摘要
    abstract = "。".join([sentences[i] for i, _ in result])

    return abstract

def generate_all_abstracts(tf_idf_dict, corpus, top_k = 3):
    res = []
    for index, document_tf_idf in tf_idf_dict.items():
        # print(corpus[index])
        title, document = corpus[index].split('\n')
        abstract = generate_document_abstracts(document_tf_idf, document, top_k)
        if abstract is None:
            abstract = document
        res.append({'标题': title, '摘要': abstract})
    return res

if __name__ == '__main__':
    json_corpus_path = "TF_IDF/corpus/"
    tf_idf = TF_IDF.TFIDF(json_corpus_path)
    tf_idf.load_corpus_from_json(json_corpus_path)
    tf_idf.build_tf_idf()
    res = generate_all_abstracts(tf_idf.tf_idf_dict, tf_idf.corpus)
    for item in res:
        print(item)
