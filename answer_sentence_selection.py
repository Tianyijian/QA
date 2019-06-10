from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn import preprocessing
from scipy.linalg import norm
import json
import const
import jieba
import numpy as np
import distance
import gensim


def gen_data():
    """ 生成训练和测试数据
    train:4353 dev:1087
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
    # 读入停用词
    stop_words = set()
    with open(const.ass_stop_words, 'r', encoding='utf-8') as f:
        for word in f.readlines():
            stop_words.add(word.strip())
    # 建立特征向量
    sents = []
    for item in items:
        sents += passage[item['pid']]
    # tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=5, stop_words=stop_words)
    # tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stop_words)
    data_feature = tv.fit_transform(sents)
    # cnt = 0
    # for key, value in tv.vocabulary_.items():
    #     print(key, value)
    #     cnt += 1
    #     if cnt >= 10: break
    print("q:{}, s:{}, avg(s/q):{}, vocab:{}".format(len(items), len(sents), float(len(sents)) / len(items),
                                                     len(tv.vocabulary_.items())))
    exit(0)
    data = []
    data_qid = []
    i = 0
    train_index = int(0.8 * float(len(items)))
    flag = False
    # print(len(items))
    # print(train_index)
    for k in range(len(items)):
        item = items[k]
        answer_sentence = ' '.join(jieba.cut(item['answer_sentence'][0]))
        # print(answer_sentence)
        for sent in passage[item['pid']]:
            # print(sent)
            feature_array = data_feature[i].toarray()[0]
            feature = ["{}:{}".format(j + 1, feature_array[j]) for j in range(len(feature_array)) if
                       feature_array[j] != 0]
            # feature = []
            # print(' '.join(feature))
            if answer_sentence == sent:
                # data.append("{} qid:{} {} #{}{}\n".format(1, item['qid'], ' '.join(feature), sents[i], sent))
                data.append("{} qid:{} {}\n".format(1, item['qid'], ' '.join(feature)))
                # data.append("{}\n".format(1))
            else:
                data.append("{} qid:{} {}\n".format(0, item['qid'], ' '.join(feature)))
                # data.append("{}\n".format(0))
            data_qid.append(item['qid'])
            i += 1
        # 写入训练集文件
        if k >= train_index - 1 and flag is False:
            # print(len(data))
            # print(k)
            sort_index = np.argsort(data_qid)
            with open(const.ass_train_data, 'w', encoding='utf-8') as f:
                for j in range(len(sort_index)):
                    f.write(data[sort_index[j]])
            flag = True
            data.clear()
            data_qid.clear()
    # 写入开发集
    # print(len(data))
    sort_index = np.argsort(data_qid)
    with open(const.ass_dev_data, 'w', encoding='utf-8') as f:
        for j in range(len(sort_index)):
            f.write(data[sort_index[j]])


def gen_data_feature():
    """生成训练和测试数据
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
        feature = ["{}:{}".format(j + 1, feature_array[j]) for j in range(0, len(feature_array))]
        # feature = ["{}:{}".format(1, feature_array[8])]
        data.append("{} qid:{} {}\n".format(item['label'], item['qid'], ' '.join(feature)))
        data_qid.append(item['qid'])
    # 写入开发集
    sort_index = np.argsort(data_qid)
    with open(const.ass_dev_data, 'w', encoding='utf-8') as f:
        for j in range(len(sort_index)):
            f.write(data[sort_index[j]])


def build_feature():
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
    # 建立词典
    sents = []
    for item in items:
        sents += passage[item['pid']]
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    cv.fit(sents)
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tv.fit(sents)
    print("q:{}, s:{}, avg(s/q):{}, vocab:{}".format(len(items), len(sents), float(len(sents)) / len(items),
                                                     len(tv.vocabulary_.items())))
    # 读取词向量
    # model = gensim.models.KeyedVectors.load_word2vec_format(const.model_file, binary=False, limit=100)
    # 建立特征矩阵
    feature = []
    sents_json = []
    for k in range(len(items)):
        item = items[k]
        answer_sentence = [' '.join(jieba.cut(ans)) for ans in item['answer_sentence']]
        for sent in passage[item['pid']]:
            # print(sent)
            # feature_array = extract_feature(item['question'], sent, cv, tv, model)
            # feature.append(' '.join([str(attr) for attr in feature_array]) + '\n')
            sen = {}
            if sent in answer_sentence:
                sen['label'] = 1
            else:
                sen['label'] = 0
            sen['qid'] = item['qid']
            sen['question'] = item['question']
            sen['answer'] = sent
            sents_json.append(sen)
    # 特征写入文件
    with open(const.ass_feature, 'w', encoding='utf-8') as f:
        f.writelines(feature)
    # 句子写入文件
    with open(const.ass_sent, 'w', encoding='utf-8') as fout:
        for sample in sents_json:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')


def extract_feature(question, answer, cv, tv, model):
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
    feature.append(cosine_similar if not np.isnan(cosine_similar) else 1.0)
    # 特征8：jaccard 系数
    numerator = np.sum(np.min(vectors, axis=0))  # 求交集
    denominator = np.sum(np.max(vectors, axis=0))  # 求并集
    feature.append(1.0 * numerator / denominator)
    # 特征9：tf-idf 相似度
    vectors = tv.transform([' '.join(question_words), answer]).toarray()
    tf_sim = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    feature.append(tf_sim if not np.isnan(tf_sim) else 1.0)
    # 特征10：word2vec 相似度
    w2v_smi = w2v_similarity(question_words, answer_words, model)
    feature.append(w2v_smi)
    return feature


def expand_feature():
    """ 在特征文件中进一步加入特征"""
    # 读取train json文件
    with open(const.train_data, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    # 读入passage json文件
    passage = {}
    with open(const.passages_seg, encoding='utf-8') as fin:
        for line in fin.readlines():
            read = json.loads(line.strip())
            passage[read['pid']] = read['document']
    # 读入停用词
    stop_words = set()
    with open(const.ass_stop_words, 'r', encoding='utf-8') as f:
        for word in f.readlines():
            stop_words.add(word.strip())
    # 建立特征向量
    sents = []
    for item in items:
        sents += passage[item['pid']]
    # tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=5, stop_words=stop_words)
    tv.fit(sents)
    # 读取特征文件
    with open(const.ass_feature, 'r', encoding='utf-8') as f:
        feature_mat = [line.strip().split(' ') for line in f.readlines()]
    # 读取sent json文件
    with open(const.ass_sent, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]


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


if __name__ == '__main__':
    # gen_data()
    # test()
    calc_mrr()
    # build_feature()
    # gen_data_feature()
    # build__test_feature()
    # gen_test_data()
