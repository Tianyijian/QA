import logging
logging = logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='result/log.txt', level=logging.DEBUG, format=LOG_FORMAT)

# 原始数据
passages_data = 'data/passages_multi_sentences.json'
test_data = 'data/test.json'
train_data = 'data/train.json'

# 生成数据
passages_seg = 'result/passages_seg.json'
index_dir = 'result/index'
index_name = 'index_file'
log_file = 'result/log.txt'
