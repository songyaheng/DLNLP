import numpy as np
import re
import xlrd
import csv


def load_data(pathpos, pathnag):
    x_text = []
    y_lable = []
    sentence_max_len = 0
    with open(pathpos, "r", encoding="utf-8") as f:
        for line in f.readlines():
            lines = line.split("=>")
            text = lines[len(lines)-1]
            y_lable.append([0, 1])
            sentence_text, sentence_len = sentence_process(text.strip())
            x_text.append(sentence_text)
            if sentence_max_len < sentence_len:
                sentence_max_len = sentence_len
    with open(pathnag, "r", encoding="utf-8") as f:
        for line in f.readlines():
            lines = line.split("=>")
            text = lines[len(lines)-1]
            y_lable.append([1, 0])
            sentence_text, sentence_len = sentence_process(text.strip())
            x_text.append(sentence_text)
            if sentence_max_len < sentence_len:
                sentence_max_len = sentence_len
    return [x_text, np.array(y_lable), sentence_max_len]


def load_predict_data(path, max_len):
    p_text = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            lines = line.split("=>")
            text = lines[len(lines)-1]
            if (len(text) > max_len):
                text = text.startswith(0, max_len - 1)
            else:
                pass
            p_text.append(text)

def sentence_process(sentence):
    #过滤非中文的词
    sentences = [w for w in sentence.split(" ") if re.match("[\u4e00-\u9fa5]+", w)]
    # sentences = [w for w in sentence.split(" ") if re.match("[a-z|A-Z]+", w)]
    #只过滤出单词长度大于等于2的（简单起见为了去掉无效词，例如“的”，“和”，“在”等）
    # sentences = [w for w in sentences if len(w) >= 2]
    sentence_len = len(sentences)
    sentence = " ".join(sentences)
    return [sentence, sentence_len]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def loadItemData(path):
    listsX = ["amount_180days",
              "amount_30days",
              "amount_7days",
              "detail_uv_rate",
              "gmv_180days",
              "gmv_30days",
              "gmv_7days",
              "transfer_rate_30days",
              "transfer_rate_7days"]
    dataX = []
    labels = []
    ExcelFile = xlrd.open_workbook(path)
    sheet = ExcelFile.sheet_by_name('Sheet0')
    num = sheet.nrows

    for i in range(1, num):
        rows = sheet.row_values(i)
        id = rows[0]
        map = str(rows[1]).lstrip("'").rstrip("'")
        label = float(rows[2])
        lb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for m in map.split(","):
            kv = m.strip().split("->")
            k = kv[0].strip()
            v = float(kv[1].strip())
            if k in listsX:
                lb[listsX.index(k)] = v
            else:
                pass
        # if label == 1.0:
        #     labels.append([1, 0])
        # else:
        #     labels.append([0, 1])
        labels.append([label])
        dataX.append(lb)
    return np.array(dataX), np.array(labels)

def loadSpuSearchAndTotal(path):
    dic = {}
    with open(path, "r") as f:
        read = csv.reader(f)
        for row in read:
            line = row[0].split("\t")
            k = line[0].strip()
            v = int(line[1].strip())
            if dic.__contains__(k):
                dic[k] = dic[k] + v
            else:
                dic[k] = v
    return dic

def loadItemData2(path, dic):
    listsX = ["amount_180days",
              "amount_30days",
              "amount_7days",
              "detail_uv_rate",
              "gmv_180days",
              "gmv_30days",
              "gmv_7days",
              "transfer_rate_30days",
              "transfer_rate_7days"]
    limit = [20000, 20000, 2000, 0.03, 0.4, 0.2, 500000, 0.4, 0.4]
    limit2 = [0, 0, 0, 0, 0.0, 0.0, 0, 0.0, 0.0]
    dataX = []
    labels = []
    ExcelFile = xlrd.open_workbook(path)
    sheet = ExcelFile.sheet_by_name('data')
    num = sheet.nrows
    for i in range(1, num):
        rows = sheet.row_values(i)
        id = str(int(rows[0]))
        map = str(rows[1]).lstrip("'").rstrip("'")

        if dic.__contains__(id):
            label = dic.get(id)
            lb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ff = [0, 0, 0, 0, 0, 0, 0, 0 ,0]
            for m in map.split(","):
                kv = m.strip().split("->")
                k = kv[0].strip()
                v = float(kv[1].strip())
                if k in listsX:
                    lv = limit[listsX.index(k)]
                    lv2 = limit2[listsX.index(k)]
                    if v < lv and v > lv2:
                        lb[listsX.index(k)] = v
                    else:
                        ff[listsX.index(k)] = 1
                        pass
                else:
                    pass
            if np.array(lb).sum() == 0:
                dataX.append(lb)
                labels.append(label)

        else:
            pass

        # if label == 1.0:
        #     labels.append([1, 0])
        # else:
        #     labels.append([0, 1])

    return np.array(dataX), np.array(labels)