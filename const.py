import logging

logging = logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(filename='./result/log.txt', level=logging.DEBUG, format=LOG_FORMAT)
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

# w2v
# model_file = '/home/syq/tyj/Test/sgns.baidubaike.bigram-char/sgns.baidubaike.bigram-char'
model_file = 'D:\BaiduNetdiskDownload\sgns.baidubaike.bigram-char\sgns.baidubaike.bigram-char'

# 原始数据
passages_data = 'data/passages_multi_sentences.json'
test_data = 'data/test.json'
train_data = 'data/train.json'
stop_words = 'question_classification/stopwords.txt'
qc_train_data = 'question_classification/train_questions.txt'
qc_test_data = 'question_classification/test_questions.txt'

# 中间数据
qc_train_seg = 'question_classification/train_questions_seg.txt'
qc_test_seg = 'question_classification/test_questions_seg.txt'
qc_train_seg_jie_ba = 'question_classification/train_questions_seg_jie_ba.txt'
qc_test_seg_jie_ba = 'question_classification/test_questions_seg_jie_ba.txt'
qc_train_rm_sw = 'question_classification/train_questions_rm_sw.txt'
qc_test_rm_sw = 'question_classification/test_questions_rm_sw.txt'
qc_train_pos = 'question_classification/train_questions_pos.txt'
qc_test_pos = 'question_classification/test_questions_pos.txt'
qc_train_ner = 'question_classification/train_questions_ner.txt'
qc_test_ner = 'question_classification/test_questions_ner.txt'
qc_train_dr = 'question_classification/train_questions_dr.txt'
qc_test_dr = 'question_classification/test_questions_dr.txt'
qc_train_rough_res = 'question_classification/train_rough_res.txt'
qc_test_rough_res = 'question_classification/test_rough_res.txt'
qc_train_fine_res = 'question_classification/train_fine_res.txt'
qc_test_fine_res = 'question_classification/test_fine_res.txt'
qc_model_rough = 'question_classification/qc_model_rough'
qc_model_fine = 'question_classification/qc_model_fine'
tv_model = 'question_classification/tv_model'

ass_train_data = 'answer_sentence_selection/ass_train.txt'
ass_dev_data = 'answer_sentence_selection/ass_dev.txt'
ass_test_data = 'answer_sentence_selection/ass_test.txt'
ass_stop_words = 'answer_sentence_selection/stopwords(new).txt'
ass_prediction = 'C:/Users/26241/Desktop/svm_rank_windows/answer_sentence_selection/predictions'
ass_feature = 'answer_sentence_selection/ass_feature.txt'
ass_test_feature = 'answer_sentence_selection/ass_test_feature.txt'
ass_sent = 'answer_sentence_selection/ass_sent.txt'
ass_test_sent = 'answer_sentence_selection/ass_test_sent.txt'

aps_train_ans = 'answer_span_selection/train_ans.json'
aps_train_diff = 'answer_span_selection/train_diff.json'

# 生成数据
passages_seg = 'result/passages_seg.json'
index_dir = 'result/index'
index_name = 'index_file'
log_file = 'result/log.txt'
test_search_res = 'result/test_res/test_search_res.json'
