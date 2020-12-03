from googletrans import Translator
import pandas as pd

df=pd.read_csv("trans2.csv",encoding="shift-jis")
name=df["name"].values
sent=df["sentence"].values

trans=Translator(service_urls=['translate.googleapis.com'])

tmp=[]
total=[]
for i in range(len(sent)):
    while True:
        try:
            transed=trans.translate(sent[i],dest="ja")
            break
        except Exception as e:
            trans=Translator(service_urls=['translate.googleapis.com'])
    tmp.append(name[i])
    tmp.append(sent[i])
    tmp.append(transed.text)
    total.append(tmp)
    tmp=[]
dft=pd.DataFrame(total)
dft.columns=["name","English","日本語"]
dft.to_csv("trans2_result.csv",encoding="shift-jis",index=False)
