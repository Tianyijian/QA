from preprocessed import *


def test_search():
    """对 test.json 文件进行搜索，获得answer_pid"""
    # 读取json文件
    with open(const.test_data, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    # 打开索引文件
    index = open_dir(const.index_dir, indexname=const.index_name)
    start = time.time()
    with index.searcher() as searcher:
        # TODO 采用Or操作，可优化
        parser = QueryParser("passage", index.schema, group=syntax.OrGroup)
        res = []
        i = 0
        for item in items:
            # q = ltp_seg_sent(item['question'])
            q = ' '.join(jieba.cut(item['question']))
            # print(q)
            results = searcher.search(parser.parse(q))
            if len(results) > 0:
                item['answer_pid'] = [int(res['id']) for res in results[0:3]]
                # pid_pre.append([int(results[0]['id'])])
            else:
                item['answer_pid'] = []
            i += 1
            res.append(item)
    end = time.time()
    print("Search {}, use time {}s".format(i, end - start))
    # 写回json文件
    with open(const.test_search_res, 'w', encoding='utf-8') as fout:
        for sample in res:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    test_search()
