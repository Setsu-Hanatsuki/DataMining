import csv
import numpy as np
from janome.tokenizer import Tokenizer

#極性辞書の作成
f=open("PNtable.csv","r",encoding="shift-jis")
reader=csv.reader(f)
dic={}
for row in reader:
    dic[row[0]]=float(row[3])
f.close()

#文書の読み込み
f=open("rashomon.txt","r",encoding="shift-jis")
text=f.read()

#形態素解析
t=Tokenizer()
word=t.tokenize(text,wakati=True)

#極性分析
val=[]
pos=[]
neg=[]
for i in range(len(word)):
    try:
        val.append(dic[word[i]])
        if dic[word[i]]<0:
            neg.append(dic[word[i]])
        elif dic[word[i]]>0:
            pos.append(dic[word[i]])
    except:
        a=1

#配列化
val=np.array(val)
pos=np.array(pos)
neg=np.array(neg)

#ポジティブ度合い
print("Positive Score")
print(np.mean(pos))

#ネガティブ度合い
print("Negative Score")
print(np.mean(neg))

#総スコア
print("Overall Score")
print(np.mean(val))
