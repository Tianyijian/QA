# -*-coding=UTF-8-*-

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import jieba
import const
import os


def ltp_seg_sent(sent):
    """
    使用LTP 对单句话进行分词
    :return:
    """
    words = segmentor.segment(sent)
    return ' '.join(words)


def ltp_seg_data():
    """使用LTP 对文本进行分词，写入文件"""
    LTP_DATA_DIR = 'D:\BaiduNetdiskDownload\ltp_data_v3.4.0'  # ltp模型目录的路径
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

    from pyltp import Segmentor

    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    result = []
    # 读入训练文件，进行分词，写回文件
    with open(const.qc_train_data, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            attr = line.strip().split('\t')
            result.append("{}\t{}\n".format(attr[0], ' '.join(segmentor.segment(attr[1]))))
    with open(const.qc_train_seg, 'w', encoding='utf-8') as f:
        f.writelines(result)
    result.clear()
    # 读入测试文件，进行分词，写回文件
    with open(const.qc_test_data, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            attr = line.strip().split('\t')
            result.append("{}\t{}\n".format(attr[0], ' '.join(segmentor.segment(attr[1]))))
    with open(const.qc_test_seg, 'w', encoding='utf-8') as f:
        f.writelines(result)
    segmentor.release()  # 释放模型


def ltp_pos_data():
    """使用 LTP 进行词性标注"""
    LTP_DATA_DIR = 'D:\BaiduNetdiskDownload\ltp_data_v3.4.0'  # ltp模型目录的路径
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

    from pyltp import Postagger
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    result = []
    file = [(const.qc_train_seg, const.qc_train_pos), (const.qc_test_seg, const.qc_test_pos)]
    for i in range(2):
        with open(file[i][0], 'r', encoding='utf-8') as f:
            for line in f.readlines():
                attr = line.strip().split('\t')
                words = attr[1].split(" ")
                words_pos = postagger.postag(words)
                res = ' '.join(["{}/_{}".format(words[i], words_pos[i]) for i in range(len(words))])
                result.append("{}\t{}\n".format(attr[0], res))
        with open(file[i][1], 'w', encoding='utf-8') as f:
            f.writelines(result)
        result.clear()
    postagger.release()  # 释放模型


def ltp_ner_data():
    """使用 LTP 进行命名实体识别"""
    LTP_DATA_DIR = 'D:\BaiduNetdiskDownload\ltp_data_v3.4.0'  # ltp模型目录的路径
    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`

    from pyltp import NamedEntityRecognizer
    recognizer = NamedEntityRecognizer()  # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型

    result = []
    file = [(const.qc_train_pos, const.qc_train_ner), (const.qc_test_pos, const.qc_test_ner)]
    for i in range(2):
        with open(file[i][0], 'r', encoding='utf-8') as f:
            for line in f.readlines():
                attr = line.strip().split('\t')
                words_pos = attr[1].split(" ")
                words = [word.split('/_')[0] for word in words_pos]
                postags = [word.split('/_')[1] for word in words_pos]
                netags = recognizer.recognize(words, postags)  # 命名实体识别
                res = ' '.join(["{}/_{}".format(words[i], netags[i]) for i in range(len(words))])
                result.append("{}\t{}\n".format(attr[0], res))
        with open(file[i][1], 'w', encoding='utf-8') as f:
            f.writelines(result)
        result.clear()
    recognizer.release()  # 释放模型


def ltp_parser_data():
    """LTP 依存句法分析提取关键词"""
    LTP_DATA_DIR = 'D:\BaiduNetdiskDownload\ltp_data_v3.4.0'  # ltp模型目录的路径
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

    from pyltp import Parser
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型

    file = [(const.qc_train_pos, const.qc_train_dr), (const.qc_test_pos, const.qc_test_dr)]
    for k in range(2):
        result = []
        with open(file[k][0], 'r', encoding='utf-8') as f:
            for line in f.readlines():
                attr = line.strip().split('\t')
                words_pos = attr[1].split(" ")
                words = [word.split('/_')[0] for word in words_pos]
                # print(words)
                postags = [word.split('/_')[1] for word in words_pos]

                arcs = parser.parse(words, postags)  # 句法分析
                # 提取核心词及其指向词
                res = []
                for i in range(len(arcs)):
                    if arcs[i].head == 0:
                        for j in range(len(arcs)):
                            if (arcs[j].head == i + 1 or arcs[j].head == 0 or arcs[j].relation == 'ADV') and arcs[j].relation != 'WP':
                            # if (arcs[j].head == i + 1 or arcs[j].head == 0) and arcs[j].relation != 'WP':
                                res.append(j)
                # print(res)
                # 广度优先选择ATT
                for j in res:
                    for i in range(len(arcs)):
                        if arcs[i].head == j+1 and arcs[i].relation == 'ATT':
                            res.append(i)
                res = list(set(res))
                res.sort()
                result.append("{}\t{}\n".format(attr[0], ' '.join([words[i] for i in res])))
        with open(file[k][1], 'w', encoding='utf-8') as f:
            f.writelines(result)
    parser.release()  # 释放模型


def ltp_parser_data_test():
    LTP_DATA_DIR = 'D:\BaiduNetdiskDownload\ltp_data_v3.4.0'  # ltp模型目录的路径

    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    from pyltp import Segmentor
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

    from pyltp import Postagger
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
    from pyltp import Parser
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型

    sents = ['API的全称是什么？', '中国是世界上老年人口最多的国家吗', '艾滋病是如何传染的', '1999年NBA总冠军是哪支球队', 'Flash是哪个公司的产品']
    for sent in sents:
        print(sent)
        words = segmentor.segment(sent)
        print(list(words))
        postags = postagger.postag(words)
        print(list(postags))
        arcs = parser.parse(words, postags)  # 句法分析
        print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        res = []
        for i in range(len(arcs)):
            if arcs[i].head == 0:
                for j in range(len(arcs)):
                    # if (arcs[j].head == i + 1 or arcs[j].head == 0 or arcs[j].relation == 'ADV' ) and arcs[j].relation != 'WP':
                    if (arcs[j].head == i + 1 or arcs[j].head == 0  ) and arcs[j].relation != 'WP':
                        res.append(j)
        print(res)
        for j in res:
            for i in range(len(arcs)):
                if arcs[i].head == j+1 and arcs[i].relation == 'ATT':
                    res.append(i)
        res = list(set(res))
        res.sort()
        print(res)
        print(' '.join([words[i] for i in res]) + '\n')

    segmentor.release()  # 释放模型
    postagger.release()  # 释放模型
    parser.release()  # 释放模型


def remove_stop_words():
    """去除停用词"""
    with open(const.stop_words, 'r', encoding='utf-8') as f:
        stop_words = [word.strip() for word in f.readlines()]
    file = [(const.qc_train_seg, const.qc_train_rm_sw), (const.qc_test_seg, const.qc_test_rm_sw)]
    for i in range(2):
        result = []
        with open(file[i][0], 'r', encoding='utf-8') as f:
            for line in f.readlines():
                attr = line.strip().split('\t')
                words = [word for word in attr[1].split(" ") if word not in stop_words]
                result.append("{}\t{}\n".format(attr[0], ' '.join(words)))
        with open(file[i][1], 'w', encoding='utf-8') as f:
            f.writelines(result)


def jie_ba_seg():
    """使用 jie ba 进行分词"""
    file = [(const.qc_train_data, const.qc_train_seg_jie_ba), (const.qc_test_data, const.qc_test_seg_jie_ba)]
    for i in range(2):
        result = []
        with open(file[i][0], 'r', encoding='utf-8') as f:
            for line in f.readlines():
                attr = line.strip().split('\t')
                result.append("{}\t{}\n".format(attr[0], ' '.join(jieba.cut(attr[1]))))
        with open(file[i][1], 'w', encoding='utf-8') as f:
            f.writelines(result)


def load_data(size='fine'):
    """加载数据
    Parameters
    ----------
    size: 细粒度还是粗粒度，fine 细粒度（默认），rough（粗粒度）
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open(const.qc_train_seg_jie_ba, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            attr = line.strip().split('\t')
            x_train.append(attr[1])
            if size == 'rough':
                y_train.append(attr[0].split('_')[0])
            else:
                y_train.append(attr[0])
    with open(const.qc_test_seg_jie_ba, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            attr = line.strip().split('\t')
            x_test.append(attr[1])
            if size == 'rough':
                y_test.append(attr[0].split('_')[0])
            else:
                y_test.append(attr[0])
    return x_train, y_train, x_test, y_test


def train_nb():
    """ 使用朴素贝叶斯进行分类"""
    x_train, y_train, x_test, y_test = load_data(size='rough')
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)

    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_data, y_train)
    print(clf.score(test_data, y_test))


def train_lr():
    """使用逻辑回归进行分类
    BoW: 小类: 0.7847908745247149  大类: 0.8973384030418251
    """
    x_train, y_train, x_test, y_test = load_data(size='rough')
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)

    lr = LogisticRegression(C=1000)
    lr.fit(train_data, y_train)
    print(lr.score(test_data, y_test))


def grid_search_lr():
    """ 网格搜索逻辑回归最优参数 """
    lr = LogisticRegression()
    parameters = [
        {
            'C': [1, 10, 50, 100, 500, 1000, 5000, 10000],
        }]
    x_train, y_train, x_test, y_test = load_data(size='rough')
    tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)

    clf = GridSearchCV(lr, parameters, cv=10, n_jobs=10)
    clf.fit(train_data, y_train)
    means = clf.cv_results_['mean_test_score']
    params = clf.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   %r" % (mean, param))
    print("------best score -------")
    print("%f  with:   %r" % (clf.best_score_, clf.best_params_))
    best_model = clf.best_estimator_
    print("test score: {}".format(best_model.score(test_data, y_test)))


def train_svm_fine():
    """ 使用 SVM 进行小类分类 """
    x_train, y_train, x_test, y_test = load_data()
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    # tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)

    # clf = SVC(decision_function_shape='ovo')
    clf = SVC(C=100.0, gamma=0.01)
    clf.fit(train_data, y_train)
    # 保存模型
    joblib.dump(clf, const.qc_model_fine)
    test_res = clf.predict(test_data)
    # 将预测结果写入文件
    with open(const.qc_test_fine_res, 'w', encoding='utf-8') as f:
        f.writelines(["{}\t{}\t{}\n".format(y_test[i], test_res[i], x_test[i]) for i in range(len(x_test))])
    print(clf.score(test_data, y_test))


def train_svm_rough():
    """ 使用SVM 进行大类分类"""
    x_train, y_train, x_test, y_test = load_data(size='rough')
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    # tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)
    # 保存tv 模型
    joblib.dump(tv, const.tv_model)
    # clf = SVC(decision_function_shape='ovo')
    # clf = SVC(C=100.0, kernel='linear')
    clf = SVC(C=100.0, gamma=0.1)
    clf.fit(train_data, y_train)
    # 保存模型
    joblib.dump(clf, const.qc_model_rough)
    test_res = clf.predict(test_data)
    # 将预测结果写入文件
    with open(const.qc_test_rough_res, 'w', encoding='utf-8') as f:
        f.writelines(["{}\t{}\t{}\n".format(y_test[i], test_res[i], x_test[i]) for i in range(len(x_test))])
    print(clf.score(test_data, y_test))


def choose_svm():
    """SVM 网格搜索调参
    TF-IDF: 大类: 100, 0.1, 0.8912547528517111  小类: 100, 0.01, 0.7809885931558935
    BoW: 大类: 100, 0.01, 0.8935361216730038 小类:100, 0.01, 0.7505703422053231
    BoW jie ba: 大类: 100, 0.02, 0.8942965779467681 小类: 100, 0.01, 0.7422053231939163
    TF-IDF jie ba: 大类: 100, 0.1,0.8996197718631179 小类: 100, 0.01, 0.7741444866920152
    """
    svc = SVC()
    parameters = [
        {
            'C': [1, 10, 50, 100, 500, 1000],
            'gamma': ['scale', 0.001, 0.002, 0.01, 0.02, 0.1, 1],
            'kernel': ['rbf']
        },
        {
            'C': [1, 10, 50, 100, 500, 1000],
            'kernel': ['linear']
        }
    ]
    x_train, y_train, x_test, y_test = load_data(size='rough')
    # x_train, y_train, x_test, y_test = load_data()
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    # tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)

    clf = GridSearchCV(svc, parameters, cv=10, n_jobs=10)
    clf.fit(train_data, y_train)
    means = clf.cv_results_['mean_test_score']
    params = clf.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   %r" % (mean, param))
    print("------best score -------")
    print("%f  with:   %r" % (clf.best_score_, clf.best_params_))
    best_model = clf.best_estimator_
    print("test score: {}".format(best_model.score(test_data, y_test)))


if __name__ == '__main__':
    # ltp_seg_data()
    # ltp_pos_data()
    # ltp_ner_data()
    # remove_stop_words()
    # ltp_parser_data()
    # ltp_parser_data_test()
    # jie_ba_seg()
    train_nb()
    train_lr()
    train_svm_fine()
    train_svm_rough()
    # choose_svm()
    # grid_search_lr()
