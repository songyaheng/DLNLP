from gensim import corpora, models, similarities
import data.DataHelper as DataHelper

pathpos = "/Users/songyaheng/Downloads/data/contentpos.txt"
pathneg = "/Users/songyaheng/Downloads/data/contentneg.txt"

x_text, y, x_max_len = DataHelper.load_data(pathpos, pathneg)

x_text = [" ".join(filter(lambda l: len(l) >= 2, x.split(" "))) for x in x_text]

text_list = [list(line.split(" ")) for line in x_text]

text_list = [t_list for t_list in text_list if len(t_list) >= 2]

word_dict = corpora.Dictionary(text_list)  #生成文档的词典，每个词与一个整型索引值对应
corpus_list = [word_dict.doc2bow(text) for text in text_list] #词频统计，转化成空间向量格式

lda = models.ldamodel.LdaModel(corpus=corpus_list,id2word=word_dict,minimum_probability = 0.04, num_topics=10, alpha='auto')

for pattern in lda.show_topics():
    print(str(pattern))