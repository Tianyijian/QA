import logging

logging = logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='./result/log.txt', level=logging.DEBUG, format=LOG_FORMAT)

# 原始数据
passages_data = 'data/passages_multi_sentences.json'
test_data = 'data/test.json'
train_data = 'data/train.json'
stop_words = 'question_classification/stopwords.txt'
qc_train_data = 'question_classification/train_questions.txt'
qc_test_data = 'question_classification/test_questions.txt'
qc_train_seg = 'question_classification/train_questions_seg.txt'
qc_test_seg = 'question_classification/test_questions_seg.txt'
qc_train_rm_sw = 'question_classification/train_questions_rm_sw.txt'
qc_test_rm_sw = 'question_classification/test_questions_rm_sw.txt'
qc_train_pos = 'question_classification/train_questions_pos.txt'
qc_test_pos = 'question_classification/test_questions_pos.txt'
qc_train_ner = 'question_classification/train_questions_ner.txt'
qc_test_ner = 'question_classification/test_questions_ner.txt'

# 生成数据
passages_seg = 'result/passages_seg.json'
index_dir = 'result/index'
index_name = 'index_file'
log_file = 'result/log.txt'
