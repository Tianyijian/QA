# -*-coding=UTF-8-*-

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
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
    with open(const.qc_train_seg, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            attr = line.strip().split('\t')
            x_train.append(attr[1])
            if size == 'rough':
                y_train.append(attr[0].split('_')[0])
            else:
                y_train.append(attr[0])
    with open(const.qc_test_seg, 'r', encoding='utf-8') as f:
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
    """使用逻辑回归进行分类"""
    x_train, y_train, x_test, y_test = load_data(size='rough')
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)

    lr = LogisticRegression(C=1000)
    lr.fit(train_data, y_train)
    print(lr.score(test_data, y_test))


def train_svm_fine():
    """ 使用 SVM 进行小类分类 """
    x_train, y_train, x_test, y_test = load_data()
    # tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)

    # clf = SVC(decision_function_shape='ovo')
    clf = SVC(C=100.0, gamma=0.01)
    clf.fit(train_data, y_train)
    print(clf.score(test_data, y_test))


def train_svm_rough():
    """ 使用SVM 进行大类分类"""
    x_train, y_train, x_test, y_test = load_data(size='rough')
    # tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)

    # clf = SVC(decision_function_shape='ovo')
    clf = SVC(C=100.0, gamma=0.01)
    clf.fit(train_data, y_train)
    # print(clf.predict(test_data))
    print(clf.score(test_data, y_test))


def choose_svm():
    """SVM 网格搜索调参"""
    svc = SVC()
    parameters = [{
        'C': [1, 10, 50, 100, 500, 1000],
        'gamma': ['scale', 0.001, 0.002, 0.01, 0.02, 0.1, 1],
        'kernel': ['rbf']
    },
    ]
    x_train, y_train, x_test, y_test = load_data(size='rough')
    # x_train, y_train, x_test, y_test = load_data()
    # tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
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
    """
    TF-IDF: 大类: 100, 0.1, 0.8912547528517111  小类: 100, 0.01, 0.7809885931558935
    BoW: 大类: 100, 0.01, 0.8935361216730038 小类:100, 0.01, 0.7505703422053231
    """
    # ltp_seg_data()
    # train_nb()
    # train_lr()
    train_svm_fine()
    train_svm_rough()
    # choose_svm()
