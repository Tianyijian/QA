import logging
logging = logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='./result/log.txt', level=logging.DEBUG, format=LOG_FORMAT)

# 原始数据
passages_data = 'data/passages_multi_sentences.json'
test_data = 'data/test.json'
train_data = 'data/train.json'
qc_train_data = 'question_classification/train_questions.txt'
qc_test_data = 'question_classification/test_questions.txt'
qc_train_seg = 'question_classification/train_questions_seg.txt'
qc_test_seg = 'question_classification/test_questions_seg.txt'

# 生成数据
passages_seg = 'result/passages_seg.json'
index_dir = 'result/index'
index_name = 'index_file'
log_file = 'result/log.txt'
