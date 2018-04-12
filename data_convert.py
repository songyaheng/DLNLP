
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
counter = collections.Counter()
with open("/Users/songyaheng/Downloads/data.conv") as f:
    ff = True
    for line in f.readlines():
        if line.strip() == "E":
            pass
        else:
            if ff and line.startswith("M"):
                line = list(line.replace("M", "").strip())
                for w in line:
                    counter[w] += 1
                ff = False
            else:
                line = list(line.replace("M", "").strip())
                for w in line:
                    counter[w] += 1
                ff = True
sorted_word_to_cnt = sorted(
    counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
word = {}
index = 0
for w in sorted_words:
    word[w] = index
    index = index + 1

f1 = open("/Users/songyaheng/Downloads/train.from", "w", encoding="utf-8")
f2 = open("/Users/songyaheng/Downloads/train.to", "w", encoding="utf-8")
with open("/Users/songyaheng/Downloads/data.conv") as f:
    ff = True
    for line in f.readlines():
        if line.strip() == "E":
            pass
        else:
            if ff and line.startswith("M"):
                line = list(line.replace("M", "").strip())
                line = map(lambda x: str(word[x]), line)
                f1.write(" ".join(line) + "\n")
                ff = False
            else:
                line = list(line.replace("M", "").strip())
                line = map(lambda x:str(word[x]), line)
                f2.write(" ".join(line) + "\n")
                ff = True
f1.close()
f2.close()


