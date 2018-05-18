
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

# convert("/Users/songyaheng/Downloads/train-images-idx3-ubyte", "/Users/songyaheng/Downloads/train-labels-idx1-ubyte",
#         "/Users/songyaheng/Downloads/mnist_train.csv", 60000)
# convert("/Users/songyaheng/Downloads/t10k-images-idx3-ubyte", "/Users/songyaheng/Downloads/t10k-labels-idx1-ubyte",
#         "/Users/songyaheng/Downloads/mnist_test.csv", 10000)
#
import collections
from operator import itemgetter
import codecs
counter = collections.Counter()
max_len = 50
VOCAB_SIZE = 6500
VOCAB_OUTPUT = "/Users/songyaheng/Downloads/vocab.txt"
with open("/Users/songyaheng/Downloads/data.conv") as f:
    ff = True
    cl = True
    for line in f.readlines():
        if line.strip() == "E":
            pass
        else:
            if ff and line.startswith("M"):
                line = list(line.replace("M", "").strip())
                if len(line) < max_len and cl:
                    for w in line:
                        counter[w] += 1
                    ff = False
                else:
                    cl = False
            else:
                line = list(line.replace("M", "").strip())
                if cl:
                    for w in line:
                        counter[w] += 1
                    ff = True
                else:
                    cl = True

sorted_word_to_cnt = sorted(
    counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
if len(sorted_words) > VOCAB_SIZE:
    sorted_words = sorted_words[:VOCAB_SIZE]
with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")

# 读取词汇表，并建立词汇到单词编号的映射。
with codecs.open(VOCAB_OUTPUT, "r", "utf-8") as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

# 如果出现了不在词汇表内的低频词，则替换为"unk"。
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

f1 = open("/Users/songyaheng/Downloads/train.from", "w", encoding="utf-8")
f2 = open("/Users/songyaheng/Downloads/train.to", "w", encoding="utf-8")
with open("/Users/songyaheng/Downloads/data.conv") as f:
    ff = True
    cl = True
    for line in f.readlines():
        if line.strip() == "E":
            pass
        else:
            if ff and line.startswith("M"):
                line = list(line.replace("M", "").strip())
                if len(line) < max_len and cl:
                    words = line + ["<eos>"]  # 读取单词并添加<eos>结束符
                    # 将每个单词替换为词汇表中的编号
                    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
                    f1.write(out_line)
                    ff = False
                else:
                    cl = False
            else:
                line = list(line.replace("M", "").strip())
                if cl:
                    words = line + ["<eos>"]  # 读取单词并添加<eos>结束符
                    # 将每个单词替换为词汇表中的编号
                    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
                    f2.write(out_line)
                    ff = True
                else:
                    cl = True

f1.close()
f2.close()
