# DataMining
"inshi"シリーズは因子分析を行っている。REGは目的変数・説明変数が連続値の場合でRFは目的変数がカテゴリで説明変数が連続値。PCAは分布の分散を用いている。<br>
"suuryou"シリーズは説明変数がカテゴリの場合にダミー変数を用いて因子分析を行う。1は重回帰分析で2はランダムフォレスト、3はPCAで4は図示。<br>
参考文献→https://algorithm.joho.info/programming/python/pandas-quantification-methods-1/<br>
"outvalue"シリーズは外れ値の発見方法で、無印は予測値とテストデータが外れた場合のsoftmax関数の値から外れ値を見つける。2はクロスバリデーションを用いてデータ群から外れ値の場所を推定する。<br>
"cvnn.py"はCNNでは判別できなかったややこしい画像をSoftmax関数の数値とともに保存して自分の目で見るプログラムです。「waim」というフォルダを作ってください。<br>
"RNN_LSTM.py"は時系列データをRNNで予測します。参考はこちら→　https://qiita.com/sasayabaku/items/b7872a3b8acc7d6261bf<br>
"mfccpredict.py"は短い音声データに対してメル周波数ケプストラム係数を使って話者認識(誰が話しているかを認識)します。<br>
"fftpredict.py"は短い音声データに対して高速フーリエ変換のパワースペクトルを使って話者認識をします。<br>
"textpredict.py"は20のニュースの中からカテゴリを減らして(メモリの容量が理由)データを分類し、ややこしい文書を探します。<br>
"textwordimp.py"は20のニュースを分類する上でどの単語が分類に寄与しているかを数値化している。
