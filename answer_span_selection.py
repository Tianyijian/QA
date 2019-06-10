from sklearn.externals import joblib
from metric import *
import const
import json
import jieba
import os
import numpy as np



def train_test():
    """使用 LTP 进行词性标注"""
    LTP_DATA_DIR = 'D:\BaiduNetdiskDownload\ltp_data_v3.4.0'  # ltp模型目录的路径

    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    from pyltp import Postagger
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    from pyltp import Segmentor
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
    from pyltp import Parser
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型

    clf = joblib.load(const.qc_model_rough)
    tv = joblib.load(const.tv_model)
    # 读取json文件
    with open(const.train_data, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    res = []
    train_diff_res = []
    none_cnt = 0
    for item in items:
        ans = item['answer']
        sent = ' '.join(jieba.cut(item['question']))
        test_data = tv.transform([sent])
        label = clf.predict(test_data)[0]
        # print(label)
        # print("{} {}".format(label, item['question']))
        # ans_words = [word for word in jieba.cut(' '.join(item['answer_sentence']))]
        q_words = [word for word in segmentor.segment(item['question'])]
        ans_sent = ' '.join(item['answer_sentence'])
        ans_words = [word for word in segmentor.segment(ans_sent)]
        # print(ans_words)
        # print([word for word in ans_words])
        words_pos = postagger.postag(ans_words)
        # print([pos for pos in words_pos])
        arcs = parser.parse(ans_words, words_pos)  # 句法分析
        if '：' in ans_sent:
            item['answer'] = ''.join(ans_sent.split('：')[1:])
        elif label == 'HUM':  # 人物
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['nh', 'ni'])
        elif label == 'LOC':
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['nl', 'ns'])
        elif label == 'NUM':
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['m'])
        elif label == 'TIME':
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['nt'])
        elif label == 'OBJ':
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['n'])
        elif label == 'DES':
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['j'])
        else:
            # item['answer'] = ''.join(ans_words)
            item['answer'] = ''
        res.append(item)
        if item['answer'] == '':
            none_cnt += 1
        tmp = {}
        tmp['l'] = label
        tmp['q'] = item['question']
        tmp['pre'] = item['answer']
        tmp['true'] = ans
        tmp['s'] = ' '.join(ans_words)
        tmp['pos'] = [pos for pos in words_pos]
        tmp['arcs'] = " ".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)
        train_diff_res.append((label, tmp))
        # print(tmp)
    print("none_cnt: {}".format(none_cnt))
    segmentor.release()
    postagger.release()  # 释放模型
    parser.release()  # 释放模型
    # 写回json文件
    with open(const.aps_train_ans, 'w', encoding='utf-8') as fout:
        for sample in res:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
    # 写中间结果文件
    train_diff_res.sort(key=lambda item:item[0])
    with open(const.aps_train_diff, 'w', encoding='utf-8') as f:
        for i in range(len(train_diff_res)):
            f.write(json.dumps(train_diff_res[i][1], ensure_ascii=False)+'\n')


# def extract_by_pos(q_words, words, words_pos, pos):
#     res = []
#     for i in range(len(words_pos)):
#         if words_pos[i] in pos:
#             res.append(i)
#     if len(res) > 1:
#         # print(q_words)
#         dis = []
#         for i in range(len(q_words)):
#             if q_words[i] in words:
#                 # print(q_words[i])
#                 dis.append(words.index(q_words[i]))
#         # print(dis)
#         # print(res)
#         mean = []
#         for i in res:
#             mean.append(np.mean(abs(np.array(dis) - i)))
#         # print(mean)
#         # print(np.argmin(mean))
#         return words[res[np.argmin(mean)]]
#     elif len(res) == 1:
#         return words[res[0]]
#     else:
#         return ''.join(res)

def extract_by_pos(q_words, words, words_pos, pos):
    res = []
    for i in range(len(words_pos)):
        if words_pos[i] in pos:
            res.append(words[i])
    if len(res):
        return ''.join(res)
    else:
        return ''



def eval():
    # 读取json文件
    with open(const.train_data, 'r', encoding='utf-8') as fin:
        train_data = [json.loads(line.strip()) for line in fin.readlines()]
    # 读取json文件
    with open(const.aps_train_ans, 'r', encoding='utf-8') as fin:
        train_ans = [json.loads(line.strip()) for line in fin.readlines()]
    cnt = len(train_data)
    # cnt = 10
    all_prediction = []
    all_ground_truth = []
    bleu = 0.0
    p = 0.0
    r = 0.0
    f1 = 0.0
    # for i in range(len(train_ans)):
    for i in range(cnt):
        bleu += bleu1(train_ans[i]['answer'], train_data[i]['answer'])
        p_1, r_1, f1_1 = precision_recall_f1(train_ans[i]['answer'], train_data[i]['answer'])
        p += p_1
        r += r_1
        f1 += f1_1
        all_prediction.append(train_ans[i]['answer'])
        all_ground_truth.append(train_data[i]['answer'])
    em = exact_match(all_prediction, all_ground_truth)
    print("bleu1:{}, em:{}, p:{}, r:{}, f1:{}".format(bleu / cnt, em, p / cnt, r / cnt, f1 / cnt))


if __name__ == '__main__':
    train_test()
    eval()
