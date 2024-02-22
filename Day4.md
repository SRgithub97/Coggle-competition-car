# Day 4 文本索引

这一块内容属于早期NLP。所以写得比较简单

## 文本索引
文本索引的核心为构建倒排索引以实现高效的文本检索。常见的检索方法有TFIDF，BM25等。

文本检索主要是基于关键词的检索方式，可以用于大规模文本数据的快速匹配。在RAG中，可以先使用文本检索筛选出候选文档，然后在这些文档上应用语义检索。这样既提高了检索效率，又能利用语义模型提取关键词的上下文信息，提升检索效果。

### 文本索引流程

1. 文本预处理
对原始文本进行清理和规范化.
包括去除停用词、标点符号等噪声，将文本统一转为小写;单词筛选等技术。还可以采用词干化或词形还原等技术，将单词转换为基本形式，以提高搜索效率。
2. 文本索引
对文档集合进行分词，得到每个文档的词项列表，并为每个词项构建倒排列表，记录包含该词项的文档及其位置信息。
3. 文本检索
用户查询经过预处理后，与建立的倒排索引进行匹配。计算查询中每个词项的权重，并利用检索算法对文档进行排序，将相关性较高的文档排在前面。

### TFIDF

TFIDF (Term Frequency-Inverse Document Frequency) 通过计算词项在文档中的频率（TF）和在整个文档集合中的逆文档频率（IDF）衡量一个词项的重要性。


$$TF(t,d) = \frac{词项t在文档d中出现的次数}{文档d中所有词项的总数}$$

$$IDF(t)=log(\frac{文档集合中的文档总数}{包含词项t的文档数 + 1})$$

$$TFIDF=TF(t,d) \times IDF(t)$$

```{python}
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# 对提问和PDF内容进行分词
question_words = [' '.join(jieba.lcut(x['question'])) for x in questions]
pdf_content_words = [' '.join(jieba.lcut(x['content'])) for x in pdf_content]

tfidf = TfidfVectorizer()
tfidf.fit(question_words + pdf_content_words)

# 提取TFIDF
question_feat = tfidf.transform(question_words)
pdf_content_feat = tfidf.transform(pdf_content_words)

# 进行归一化
question_feat = normalize(question_feat)
pdf_content_feat = normalize(pdf_content_feat)

# 检索进行排序
for query_idx, feat in enumerate(question_feat):
    score = feat @ pdf_content_feat.T
    score = score.toarray()[0]
    max_score_page_idx = score.argsort()[-1] + 1
    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx)
```

### BM25

BM25Okapi是BM25算法的一种变体，通过score来评估检索中文档与查询之间的相关性。

$$ score = \sum_{q \in query} ( IDF(q) \cdot \frac{q_{freq} \cdot (k1 + 1)}{q_{freq} + k1 \cdot (1 - b + b \cdot \frac{doc-len}{avgdl})})
$$

其中，$IDF$ 为上面的逆文档频率， $q_{freq}$ 表示词项在文档中的频率，doc-len表示文档长度，avgdl表示文档集合中的平均文档长度。主要参数为k1,b,和epsilon。 k1表示控制词项频率对分数的影响，通常设置为1.5。b表示控制文档长度对分数的影响，通常设置为0.75。epsilon用于防止逆文档频率（IDF）为负值的情况，通常设置为0.25。

BM25Okapi通过合理调整参数，兼顾了词项频率、文档长度和逆文档频率，能更准确地评估文档与查询之间的相关性。

```{python}
# !pip install rank_bm25
from rank_bm25 import BM25Okapi

pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words)

for query_idx in range(len(questions)):
    doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))
    max_score_page_idx = doc_scores.argsort()[-1] + 1
    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx)

with open('submit.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)
```
实际运用中，这种方法比较占内存，可以考虑基于倒排索引等数据结构的文本检索引擎。但是由于本次考题所给的pdf尽管可能比较厚，但是每一次中所含有的文本内容并不多，所以这种方法还是可以的。

