#!/user/bin/env python3
"""
File_Name: es_kb_service.py
Author: JieYang
Email: 3149156597@qq.com
Created: 2024-05-27
"""

import os
import time
import shutil
from typing import List, Dict
from functools import wraps
from langchain.schema import Document
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain.embeddings.base import Embeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import sys

sys.path.append("/home/llmapi/finreport_test/")


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        print(f"Function: {func.__name__} execution time: {execution_time} ms")
        return result

    return wrapper


class ESKBService:
    def __init__(self) -> None:
        self.do_init()

    def do_init(self):
        self.kb_name = "zte2024"
        self.index_name = "zte2024"
        self.IP = "10.13.14.16"
        self.PORT = "9200"
        self.dims_length = 756
        # self.embeddings_model = HuggingFaceEmbeddings(
        #     model_name="m3e", model_kwargs={"device": "cpu"}
        # )
        self.es_client_python = Elasticsearch(f"http://{self.IP}:{self.PORT}")
        if not self.es_client_python.indices.exists(index=self.index_name):
            body = {
                "settings": {
                    "analysis": {
                        "filter": {
                            "my_stopwords_filter": {
                                "type": "stop",
                                "stopwords_path": "analysis-ik/custom/stopwords.dic",
                            },
                            "my_terminology_filter": {
                                "type": "dictionary_decompounder",
                                "word_list_path": "analysis-ik/custom/terminology.dic",
                            },
                        },
                        "analyzer": {
                            "my_analyzer": {
                                "tokenizer": "ik_smart",
                                "filter": [
                                    "my_stopwords_filter",
                                    "my_terminology_filter",
                                ],
                            }
                        },
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "my_analyzer"},
                        "dense_vector": {
                            "type": "dense_vector",
                            "dims": self.dims_length,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "metadata": {"type": "text"},
                    }
                },
            }
            response = self.es_client_python.indices.create(
                index=self.index_name, body=body
            )
            # print("索引创建成功：", response)
        else:
            # print("索引已存在，未创建新索引。")
            pass
        # print("连接到 Elasticsearch 成功！")

    def _load_es(self, docs: List[str], embed_model):
        # text_chuncks = [doc.page_content for doc in docs]
        # text_vectors = embed_model.embed_documents(text_chuncks)
        actions = [
            {
                "_index": "zte2024",
                "_source": {
                    "content": doc,
                    "metadata": "英文",
                    # "dense_vector": text_vectors[i],
                },
            }
            for i, doc in enumerate(docs)
        ]
        # print(actions)
        bulk(self.es_client_python, actions)

    @timeit
    def do_add_doc(self, docs: List[str]):
        """向知识库添加文件"""
        # print(
        #     f"server.knowledge_base.kb_service.es_kb_service.do_add_doc 输入的docs参数长度为:{len(docs)}"
        # )
        # print("*" * 100)
        self._load_es(docs=docs, embed_model="self.embeddings_model")
        # print("写入数据成功.")
        # print("*" * 100)

        return True

    @timeit
    def do_search(
        self,
        query: str,
        top_k: int = 50,
    ):
        query_body_keyword = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "analyzer": "my_analyzer",
                                },
                            },
                        },
                    ]
                }
            },
            "_source": ["content", "metadata"],
        }

        query_body = {
            **query_body_keyword,
            "size": top_k,
        }
        response = self.es_client_python.search(index=self.index_name, body=query_body)
        # print(len(response["hits"]["hits"]))
        # print(response["hits"]["hits"][0])
        docs = [hit["_source"]["content"] for hit in response["hits"]["hits"]]
        return docs

    def search_docs(
        self,
        query: str,
        company_info: Dict,
        top_k: int = 50,
    ):
        embeddings = self._load_embeddings()
        docs = self.do_search(query, top_k, company_info, embeddings)
        return docs


if __name__ == "__main__":
    # 写入数据
    esKBService = ESKBService("zte2024")
    data_list = ["1111" for i in range(10)]
    esKBService.do_add_doc(data_list)
