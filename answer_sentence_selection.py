from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn import preprocessing
from gensim.summarization import bm25
from scipy.linalg import norm
import json
import const
import jieba
import numpy as np
import distance
import gensim


def gen_data_feature():
    """生成 SVM Rank 格式的训练和测试数据
    train:4353 dev:1087
    """
    # 读取sent json文件
    with open(const.ass_sent, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    # 读取特征文件
    with open(const.ass_feature, 'r', encoding='utf-8') as f:
        feature_mat = [line.strip().split(' ') for line in f.readlines()]
    print(len(items))
    print(len(feature_mat))
    data = []
    data_qid = []
    qid_set = set()
    train_index = int(0.8 * float(5352))
    flag = False
    for k in range(len(items)):
        item = items[k]
        qid_set.add(item['qid'])
        # 写入训练集文件
        if len(qid_set) >= train_index + 1 and flag is False:
            sort_index = np.argsort(data_qid)
            with open(const.ass_train_data, 'w', encoding='utf-8') as f:
                for j in range(len(sort_index)):
                    f.write(data[sort_index[j]])
            flag = True
            data.clear()
            data_qid.clear()
        feature_array = feature_mat[k]
        index = [0, 1, 2, 3, 4, 5, 6, 8, 10]
        # feature = ["{}:{}".format(j + 1, feature_array[j]) for j in range(0, len(feature_array))]
        feature = ["{}:{}".format(j + 1, feature_array[index[j]]) for j in range(len(index))]
        # feature = ["{}:{}".format(1, feature_array[10])]
        data.append("{} qid:{} {}\n".format(item['label'], item['qid'], ' '.join(feature)))
        data_qid.append(item['qid'])
    # 写入开发集
    sort_index = np.argsort(data_qid)
    with open(const.ass_dev_data, 'w', encoding='utf-8') as f:
        for j in range(len(sort_index)):
            f.write(data[sort_index[j]])


def build_feature():
    """ 建立特征文件
    q:5352, s:112055, avg(s/q):20.93703288490284, vocab:168611
    """
    # 读取train json文件
    with open(const.train_data, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    # 读入passage json文件
    passage = {}
    with open(const.passages_seg, encoding='utf-8') as fin:
        for line in fin.readlines():
            read = json.loads(line.strip())
            passage[read['pid']] = read['document']
    # 读入raw passage json文件
    passage_raw = {}
    with open(const.passages_data, encoding='utf-8') as fin:
        for line in fin.readlines():
            read = json.loads(line.strip())
            passage_raw[read['pid']] = read['document']

    # 读取词向量
    model = gensim.models.KeyedVectors.load_word2vec_format(const.model_file, binary=False, limit=100)
    # 建立特征矩阵
    feature = []
    sents_json = []
    for k in range(len(items)):
    # for k in range(100):
        item = items[k]

        # 建立词袋模型
        cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        cv.fit(passage[item['pid']])
        tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tv.fit(passage[item['pid']])

        # 提取 BM25 特征
        corpus = []
        for sent in passage[item['pid']]:
            corpus.append(sent.split())
        # print(corpus)
        # print(len(corpus))
        bm25_model = bm25.BM25(corpus)
        average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())
        q = list(jieba.cut(item['question']))
        scores = bm25_model.get_scores(q, average_idf)

        for i in range(len(passage[item['pid']])):
            # print(sent)
            ans_sent = passage[item['pid']][i]
            feature_array = extract_feature(item['question'], ans_sent, cv, tv, model, scores[i])
            feature.append(' '.join([str(attr) for attr in feature_array]) + '\n')
            sen = {}
            if passage_raw[item['pid']][i] in item['answer_sentence']:
                sen['label'] = 1
            else:
                sen['label'] = 0
            sen['qid'] = item['qid']
            sen['question'] = item['question']
            sen['answer'] = passage[item['pid']][i]
            sents_json.append(sen)
    # 特征写入文件
    with open(const.ass_feature, 'w', encoding='utf-8') as f:
        f.writelines(feature)
    # 句子写入文件
    with open(const.ass_sent, 'w', encoding='utf-8') as fout:
        for sample in sents_json:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')


def extract_feature(question, answer, cv, tv, model, bm25_score):
    """ 抽取句子各种特征"""
    feature = []
    question_words = list(jieba.cut(question))
    answer_words = answer.split(' ')
    # print(question_words)
    # print(answer_words)
    # 特征1：答案句词数
    feature.append(len(answer_words))
    # 特征2：是否含冒号
    if '：' in answer:
        feature.append(1)
    else:
        feature.append(0)
    # 特征3：问句和答案句词数差异
    feature.append(abs(len(question_words) - len(answer_words)))
    # 特征4：编辑距离
    feature.append(distance.levenshtein(question, ''.join(answer_words)))
    # 特征5：unigram 词共现比例：答案句和问句中出现的相同词占问句总词数的比例
    feature.append(len(set(question_words) & set(answer_words)) / float(len(set(question_words))))
    # 特征6：字符共现比例:答案句和问句中出现的相同字符占问句的比例
    feature.append(len(set(question) & set(''.join(answer_words))) / float(len(set(question))))
    # 特征7：one hot 余弦相似度
    vectors = cv.transform([' '.join(question_words), answer]).toarray()
    cosine_similar = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    feature.append(cosine_similar if not np.isnan(cosine_similar) else 0.0)
    # 特征8：jaccard 系数
    numerator = np.sum(np.min(vectors, axis=0))  # 求交集
    denominator = np.sum(np.max(vectors, axis=0))  # 求并集
    feature.append(1.0 * numerator / denominator)
    # 特征9：tf-idf 相似度
    vectors = tv.transform([' '.join(question_words), answer]).toarray()
    tf_sim = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    feature.append(tf_sim if not np.isnan(tf_sim) else 0)
    # 特征10：word2vec 相似度
    w2v_smi = w2v_similarity(question_words, answer_words, model)
    feature.append(w2v_smi)
    # 特征11：bm25 评分
    feature.append(bm25_score)
    return feature


def w2v_similarity(s1, s2, model):
    """ 利用w2v计算句子相似度"""

    def sentence_vector(s):
        v = np.zeros(300)
        cnt = 0
        for word in s:
            if word in model.vocab:
                v += model[word]
                cnt += 1
        if cnt >= 1:
            v /= cnt
        return v

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    w2v_smi = np.dot(v1, v2) / (norm(v1) * norm(v2))
    return w2v_smi if not np.isnan(w2v_smi) else 0.0


def calc_mrr():
    """ 计算排序结果的 MRR """
    # 读入排序结果
    with open(const.ass_prediction, 'r', encoding='utf-8') as f:
        predictions = np.array([float(line.strip()) for line in f.readlines()])
    # print(len(predictions))
    # 读入开发集文件
    dev = []
    with open(const.ass_dev_data, 'r', encoding='utf-8') as f:
        i = 0
        for line in f.readlines():
            dev.append((i, int(line[0]), int(line.split(' ')[1].split(':')[1])))
            i += 1
    # print(len(dev))
    # 统计并计算 MRR
    old_pid = dev[0][2]
    q_s = 0
    question_num = 0
    question_with_answer = 0
    prefect_correct = 0
    mrr = 0.0
    for i in range(len(dev)):
        if dev[i][2] != old_pid:
            # print(i)
            p = np.argsort(-predictions[q_s:i]) + q_s
            for k in range(len(p)):
                if dev[p[k]][1] == 1:
                    question_with_answer += 1
                    if k == 0:
                        prefect_correct += 1
                    mrr += 1.0 / float(k + 1)
                    break
            # print(p)
            q_s = i
            old_pid = dev[i][2]
            question_num += 1
    p = np.argsort(-predictions[q_s:]) + q_s
    for k in range(len(p)):
        if dev[p[k]][1] == 1:
            question_with_answer += 1
            if k == 0:
                prefect_correct += 1
            mrr += 1.0 / float(k + 1)
            break
    question_num += 1
    print(
        "question num:{}, question with answer{}, prefect_correct:{}, MRR:{}".format(question_num, question_with_answer,
                                                                                     prefect_correct,
                                                                                     mrr / question_num))


def test():
    """ 仅用于特征抽取测试 """
    sents = ['没有 你 的 地方 都是 他乡', '没有 你 的 旅行 都是 流浪']
    tv = TfidfVectorizer()
    # tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    data_feature = tv.fit_transform(sents)
    print(data_feature.toarray())
    print(data_feature[0].toarray())
    feature_array = data_feature[0].toarray()[0]
    feature = ["{}:{}".format(i + 1, feature_array[i]) for i in range(len(feature_array))]
    print(' '.join(feature))


def build__test_feature():
    """ 建立测试文件的特征文件
    q:500, s:33530, avg(s/q):67.06, vocab:70797
    """
    # 读取test json文件
    with open(const.test_search_res, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    # 读入passage json文件
    passage = {}
    with open(const.passages_seg, encoding='utf-8') as fin:
        for line in fin.readlines():
            read = json.loads(line.strip())
            passage[read['pid']] = read['document']
    answer_pid = []
    for item in items:
        # if len(item['answer_pid']) == 2:
        #     print(item['qid'])
        for pid in item['answer_pid']:
            answer_pid.append(pid)
    print(len(answer_pid))
    # 建立词典
    sents = []
    for pid in answer_pid:
        sents += passage[pid]
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    cv.fit(sents)
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tv.fit(sents)
    print("q:{}, s:{}, avg(s/q):{}, vocab:{}".format(len(items), len(sents), float(len(sents)) / len(items),
                                                     len(tv.vocabulary_.items())))
    # 读取词向量
    model = gensim.models.KeyedVectors.load_word2vec_format(const.model_file, binary=False, limit=100)
    # 建立特征矩阵
    feature = []
    sents_json = []
    for k in range(len(items)):
        item = items[k]
        for pid in item['answer_pid']:
            for sent in passage[pid]:
                # print(sent)
                feature_array = extract_feature(item['question'], sent, cv, tv, model)
                feature.append(' '.join([str(attr) for attr in feature_array]) + '\n')
                sen = {}
                sen['label'] = 0
                sen['qid'] = item['qid']
                sen['question'] = item['question']
                sen['answer'] = sent
                sents_json.append(sen)
    # 特征写入文件
    with open(const.ass_test_feature, 'w', encoding='utf-8') as f:
        f.writelines(feature)
    # 句子写入文件
    with open(const.ass_test_sent, 'w', encoding='utf-8') as fout:
        for sample in sents_json:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')


def gen_test_data():
    """生成测试数据
    test 500
    """
    # 读取sent json文件
    with open(const.ass_test_sent, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    # 读取特征文件
    with open(const.ass_test_feature, 'r', encoding='utf-8') as f:
        feature_mat = [line.strip().split(' ') for line in f.readlines()]
    print(len(items))
    print(len(feature_mat))
    data = []
    data_qid = []
    qid_set = set()
    for k in range(len(items)):
        item = items[k]
        qid_set.add(item['qid'])
        feature_array = feature_mat[k]
        feature = ["{}:{}".format(j + 1, feature_array[j]) for j in range(0, len(feature_array))]
        # feature = ["{}:{}".format(1, feature_array[8])]
        data.append("{} qid:{} {}\n".format(item['label'], item['qid'], ' '.join(feature)))
        data_qid.append(item['qid'])
    # 写入开发集
    sort_index = np.argsort(data_qid)
    with open(const.ass_test_data, 'w', encoding='utf-8') as f:
        for j in range(len(sort_index)):
            f.write(data[sort_index[j]])


def bm25_feature():
    """ 建立特征文件 """
    # 读取train json文件
    with open(const.train_data, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    # 读入passage json文件
    passage = {}
    with open(const.passages_seg, encoding='utf-8') as fin:
        for line in fin.readlines():
            read = json.loads(line.strip())
            passage[read['pid']] = read['document']
    # 读入raw passage json文件
    passage_raw = {}
    with open(const.passages_data, encoding='utf-8') as fin:
        for line in fin.readlines():
            read = json.loads(line.strip())
            passage_raw[read['pid']] = read['document']
    # 建立词典
    sents = []
    for item in items:
        sents += passage[item['pid']]
    for k in range(len(items)):
        # for k in range(2):
        item = items[k]
        answer_sentence = [' '.join(jieba.cut(ans)) for ans in item['answer_sentence']]
        corpus = []
        for sent in passage[item['pid']]:
            corpus.append(sent.split())
        # print(corpus)
        # print(len(corpus))
        bm25_model = bm25.BM25(corpus)
        average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())
        q = list(jieba.cut(item['question']))
        scores = bm25_model.get_scores(q, average_idf)
        # print(scores)
        # print(len(scores))
        data = []
        for i in range(len(corpus)):
            if passage_raw[item['pid']][i] in item['answer_sentence']:
                label = 1
            else:
                label = 0
            data.append("{} qid:{} 1:{}\n".format(label, item['qid'], scores[i]))
        with open(const.ass_bm25_pre, 'a', encoding='utf-8') as f:
            f.writelines(data)


def calc_mrr_bm25():
    """ 计算排序结果的 MRR
    """
    # 读入排序结果
    with open(const.ass_bm25_pre, 'r', encoding='utf-8') as f:
        predictions = np.array([float(line.split(' ')[2].split(':')[1]) for line in f.readlines()])
    # print(len(predictions))
    # 读入开发集文件
    dev = []
    with open(const.ass_bm25_pre, 'r', encoding='utf-8') as f:
        i = 0
        for line in f.readlines():
            dev.append((i, int(line[0]), int(line.split(' ')[1].split(':')[1])))
            i += 1
    # print(len(dev))
    # 统计并计算 MRR
    old_pid = dev[0][2]
    q_s = 0
    question_num = 0
    question_with_answer = 0
    prefect_correct = 0
    mrr = 0.0
    for i in range(len(dev)):
        if dev[i][2] != old_pid:
            # print(i)
            p = np.argsort(-predictions[q_s:i]) + q_s
            for k in range(len(p)):
                if dev[p[k]][1] == 1:
                    question_with_answer += 1
                    if k == 0:
                        prefect_correct += 1
                    mrr += 1.0 / float(k + 1)
                    break
            # print(p)
            q_s = i
            old_pid = dev[i][2]
            question_num += 1
    p = np.argsort(-predictions[q_s:]) + q_s
    for k in range(len(p)):
        if dev[p[k]][1] == 1:
            question_with_answer += 1
            if k == 0:
                prefect_correct += 1
            mrr += 1.0 / float(k + 1)
            break
    question_num += 1
    print(
        "question num:{}, question with answer{}, prefect_correct:{}, MRR:{}".format(question_num, question_with_answer,
                                                                                     prefect_correct,
                                                                                     mrr / question_num))


if __name__ == '__main__':
    """
    TF: question num:1071, question with answer1064, prefect_correct:491, MRR:0.6200992728238546
    Bow: question num:1071, question with answer1064, prefect_correct:376, MRR:0.5296269550798569
    BM25 + SVMRank: question num:1071, question with answer1064, prefect_correct:550, MRR:0.6524602350657909
    BM25:  question num:5352, question with answer5319, prefect_correct:2783, MRR:0.6594948607696011
    -- [8, 10]: BM25 + TF: question num:1071, question with answer1064, prefect_correct:566, MRR:0.6626076288570559
    1  [0, 8, 10]: question num:1071, question with answer1064, prefect_correct:571, MRR:0.6672033276872427
    1  [2, 8, 10]: question num:1071, question with answer1064, prefect_correct:573, MRR:0.6696462509248209
    -  [6, 8, 10]: question num:1071, question with answer1064, prefect_correct:566, MRR:0.6623672110525073
    0  [9, 8, 10]: question num:1071, question with answer1064, prefect_correct:564, MRR:0.6606765867066795
    0  [7, 8, 10]: question num:1071, question with answer1064, prefect_correct:563, MRR:0.6607581853759598
    1  [5, 8, 10]: question num:1071, question with answer1064, prefect_correct:567, MRR:0.6702383345109053
    1  [4, 8, 10]: question num:1071, question with answer1064, prefect_correct:566, MRR:0.6630578089290848
    1  [3, 8, 10]: question num:1071, question with answer1064, prefect_correct:575, MRR:0.6726265375296776
    1  [3, 4, 5, 8, 10]: question num:1071, question with answer1064, prefect_correct:575, MRR:0.6774593108601733
    1  [3, 4, 5,6,8,10]: question num:1071, question with answer1064, prefect_correct:582, MRR:0.6815223646174593
       [3, 4, 5,6,7,8,10]: question num:1071, question with answer1064, prefect_correct:580, MRR:0.6808120259425803
    =  [0, 1, 2, 3, 4, 5, 6,8,10]: question num:1071, question with answer1064, prefect_correct:615, MRR:0.7118507347966923  1000
    ALL: question num:1071, question with answer1064, prefect_correct:608, MRR:0.7071495565046418  1000
    == ：[0, 1, 2, 3, 4, 5, 6,8,10]: question num:1071, question with answer1064, prefect_correct:618, MRR:0.7163615823193664 5000
    ------------not remove stopwords---------
    -- [8, 10]: BM25 + TF: question num:1071, question with answer1064, prefect_correct:500, MRR:0.622616494858178
    [0, 1, 2, 3, 4, 5, 6,8,10]: question num:1071, question with answer1064, prefect_correct:604, MRR:0.7060185519102772  1000
    ALL: question num:1071, question with answer1064, prefect_correct:605, MRR:0.7077294225814748
    """
    # gen_data()
    # test()
    calc_mrr()
    # build_feature()
    # gen_data_feature()
    # build__test_feature()
    # gen_test_data()
    # bm25_feature()
    # calc_mrr_bm25()
