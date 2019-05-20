# -*-coding=UTF-8-*-
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.analysis import KeywordAnalyzer
from whoosh.qparser import syntax
import numpy as np
import json
import os
import time
import const

LTP_DATA_DIR = 'D:\BaiduNetdiskDownload\ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

from pyltp import Segmentor

segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型


def ltp_seg():
    """
    使用LTP 对json文件中的标题和正文进行分词
    :return:
    """
    LTP_DATA_DIR = 'D:\BaiduNetdiskDownload\ltp_data_v3.4.0'  # ltp模型目录的路径
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

    from pyltp import Segmentor
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    # 读入json文件
    read_results = []
    with open(const.passages_data, encoding='utf-8') as fin:
        read_results = [json.loads(line.strip()) for line in fin.readlines()]
    # 对正文进行分词
    start = time.time()
    for res in read_results:
        res['document'] = [' '.join(segmentor.segment(sent)) for sent in res['document']]
    end = time.time()
    # 写回json文件
    with open(const.passages_seg, 'w', encoding='utf-8') as fout:
        for sample in read_results:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
    segmentor.release()  # 释放模型
    print("LTP seg done, use time: {}s".format(end - start))  # LTP seg done, use time: 121.8924994468689s


def ltp_seg_sent(sent):
    """
    使用LTP 对单句话进行分词
    :return:
    """
    words = segmentor.segment(sent)
    return ' '.join(words)


def create_index():
    # 定义索引域 TODO 采用KeywordAnalyzer,可优化
    schema = Schema(id=ID(stored=True), passage=TEXT(stored=False))
    # schema = Schema(id=ID(stored=True), passage=TEXT(stored=False, analyzer=KeywordAnalyzer()))
    ix = create_in(const.index_dir, schema, indexname=const.index_name)
    writer = ix.writer()
    # 读取json文件
    with open(const.passages_seg, 'r', encoding='utf-8') as fin:
        docs = [json.loads(line.strip()) for line in fin.readlines()]
    # 加入索引
    start = time.time()
    for doc in docs:
        # print(''.join(doc['document']))
        writer.add_document(id=str(doc['pid']), passage=' '.join(doc['document']))
    writer.commit()
    end = time.time()
    print("Create index done, use time {}s".format(end - start))


def search():
    index = open_dir(const.index_dir, indexname=const.index_name)
    with index.searcher() as searcher:
        # TODO 采用Or操作，可优化
        parser = QueryParser("passage", index.schema, group=syntax.OrGroup)
        query = parser.parse("腾讯 在线 教育 由 哪 几个 部分 组成")
        results = searcher.search(query)
        for hit in results:
            print(hit)


def train_test():
    # 读取json文件
    with open(const.train_data, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    # 打开索引文件
    index = open_dir(const.index_dir, indexname=const.index_name)
    start = time.time()
    time1 = time.time()
    with index.searcher() as searcher:
        # TODO 采用Or操作，可优化
        parser = QueryParser("passage", index.schema, group=syntax.OrGroup)
        pid_label = []
        pid_pre = []
        i = 0
        # for item in items[0:500]:
        for item in items:
            pid_label.append(item['pid'])
            q = ltp_seg_sent(item['question'])
            # print(q)
            results = searcher.search(parser.parse(q))
            if len(results) > 0:
                pid_pre.append([int(res['id']) for res in results[0:3]])
                # pid_pre.append([int(results[0]['id'])])
            else:
                pid_pre.append([])
            i += 1
            if i % 1000 == 0:
                const.logging.debug("search {} done, use time {}s".format(i, time.time() - time1))
                time1 = time.time()
    end = time.time()
    print("Search {}, use time {}s".format(i, end - start))
    # print(pid_label)
    # print(pid_pre)
    const.logging.debug("Search {}, use time {}s".format(i, end - start))
    const.logging.debug(pid_label[0:3])
    const.logging.debug(pid_pre[0:3])
    eval(pid_label, pid_pre)


def eval(label, pre):
    rr_rn = len(label)  # 检索回来的文档总数 按问题数目算
    rr = 0  # 检索回来的相关文档数
    rr_nr = len(label)  # 相关文档总数
    for i in range(len(label)):
        # rr_rn += len(pre[i])
        if label[i] in pre[i]:
            rr += 1
    p = float(rr) / rr_rn
    r = float(rr) / rr_nr
    f = 2 * p * r / (p + r)
    print("Total:{}, rr:{}, rr_rn:{}, rr_nr:{}, P:{}, R:{}, F1:{}".format(len(label), rr, rr_rn, rr_nr, p, r, f))
    const.logging.debug(
        "Total:{}, rr:{}, rr_rn:{}, rr_nr:{}, P:{}, R:{}, F1:{}".format(len(label), rr, rr_rn, rr_nr, p, r, f))


if __name__ == '__main__':
    # ltp_seg()
    # create_index()
    # search()
    train_test()
