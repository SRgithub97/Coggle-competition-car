# Day 3 读取数据

本次数据来源于[2023全球智能汽车AI挑战赛——赛道一：AI大模型检索问答](https://tianchi.aliyun.com/competition/entrance/532154)。

数据包含两个部分，一是数据集，另一个是测试。数据集为pdf格式，测试内容为json格式。

## 读取数据集

```{python}
#使用pdfplumber
import pdfplumber
pdf = pdfplumber.open("初赛训练数据集.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text()
    })

```

```{python}
#使用langchain读取pdf数据
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("初赛训练数据集.pdf")
data=loader.load()
#这边会生成langchain的Document类型文档。
```

