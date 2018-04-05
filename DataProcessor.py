from data.DataHelper import loadSpuSearchAndTotal

total = loadSpuSearchAndTotal("/Users/songyaheng/Downloads/total.csv")
search = loadSpuSearchAndTotal("/Users/songyaheng/Downloads/search.csv")

m = {}
for k in search:
    sv = search.get(k)
    tv = total.get(k)
    v = 0
    if tv == 0:
        m[k] = v
    else:
        m[k] = sv / tv
print(m)

with open("/Users/songyaheng/Downloads/searchandtotal.txt", "w", encoding="utf-8") as f:
    for k in m:
        v = m.get(k)
        f.write(k + " " + str(v) + "\n")