# 为什么 LSTM 检测 DGA 是无用功

本文的所有内容与作者的日常工作无关，其观点也仅代表作者个人意见，与作者的雇主无关。

基于字符集特征检测 DGA 的 LSTM 模型的判别效果在对抗样本下十分脆弱，其在固定数据集上表现的 99.9% 准确率在海量 DNS 数据中的误判会带来很高运营成本，使得在生产系统里部署这类模型变得十分不明智。LSTM 在该问题上产生无用功，其背后的原因来自于捷径学习。本文从探究“LSTM 检测 DGA 为什么是错误的”出发，简单介绍基于样本的对抗以及一些“捷径学习”的分析，文中提到的论文和开源代码均在参考文献部分列出供各位小伙伴们参考。

## 字符集特征检测 DGA

从 DNS 流量日志出发检测域名生成算法 （DGA）产生的僵尸网络中控域名一直是个热门话题，其方法一般分为利用域名字符集特征的判别模型以及利用 DNS 流量序列/共现特征的聚类两个方向，而第一个方向因为更容易做成在线模型更受欢迎，同时它也不太需要为当前 DNS 日志更新模型，对于安全盒子类不易回传 DNS 日志的场景更加友好。

基于字符集特征的判别模型利用了 DGA 域名的字符特性，比如频率和位置的随机性、元音辅音的混合比例、n-gram的组合等，通常使用已知 DGA 当作正例 Alexa Top 域名当作反例，训练并输出二分类或者多分类模型。业界最早用到产品系统里的案例来自于 CrowdStrike 在 BlackHat 2013 的报告 “End-to-End Analysis of a Domain Generating Algorithm Malware Family”； 2015 年本文作者在乌云知识库里分享一篇利用 SVM 检测 DGA 的文章“用机器学习识别随机生成的C&C域名”并开源了代码；到了 2018 年，FANCI 在 usenixsecurity18 的文章 “FANCI : Feature-based Automated NXDomain Classification and Intelligence”
和相关开源代码被业界接受并认可。这些判别模型利用了基于统计的特征工程，不同作者有不同的巧思，也在各自的数据集里达到了99%以上的结果。

然后深度神经网络热起来了，有聪明的小伙伴提出，不如让深度神经网络自己学习这些特征，这不机智么？

### LSTM 看起来更加“深度学习”

在 2016 年底，Endgame （现在改名叫 Elastic）的研究人员构建了一个简单的 LSTM，利用域名里相邻字符出现的顺序作为输入并构建判别模型；在 2018 年 Duc Tran 等的文章 “A LSTM based framework for handling multiclass imbalance in DGA botnet detection” 提出了对非平衡分类的改进，其开源的代码 LSTM.MI 也被业界多次引用。它们都基于这个简单到不能再简单的 LSTM 网络结构（节选自 endgame 的开源实现代码，LSTM.MI 与此结构完全一致）：

```
def build_model(max_features, maxlen):
    """Build LSTM model"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')

    return model
```
熟悉 LSTM 特性的小伙伴们可以看出它和常见做文本分类教程里的 LSTM 例子几乎一样，甚至连编码向量的长度 128 都没有变化，它学习到的特征可以认为是 n-gram 类特征对于该数据集的一个子集。那么，更加聪明的小伙伴提出，对于这个白盒模型，我们能不能攻击一下呢？

其实有效攻击方法远比各位小伙伴想象的简单，而这些 LSTM 模型比各位想象的更脆弱。

### 针对 LSTM 类 DGA 检测的简单攻击

针对 LSTM DGA 检测的攻击，有知道训练数据集或网络结构的白盒攻击和相对应的黑盒办法。 

“CharBot: A Simple and Effective Method for Evading DGA Classifiers” 是白盒攻击法的一种，其方法简单到甚至不需要任何代码：从训练数据集的反例里随便找一个域名并选取两个随机位置替换为另外两个随机字符，再随机替换其 TLD，这样的简单的攻击可将 LSTM.MI 等模型准确率降低到 90% 而检测率甚至可以低到 30% 左右。

一些黑盒攻击的模型也比较有意思，比如 “MaskDGA: A Black-box Evasion Technique Against DGA Classifiers and Adversarial Defenses” 通过简单的对抗生成得到一个很浅的网络，已经可以 LSTM.MI 等模型的 F-1 score 从 0.977 降到 0.495，变得和瞎猜没区别。类似的做法比如 DeepDGA，DeceptionDGA 等也遵循类似方法。

由此可见，LSTM 类 DGA 检测面对简单的基于样本的攻击表现的很脆弱，更不用说其声称的 99.9% 准确率在验证数据集之外的大规模 DNS 流量日志里出现的诸多域名都有可能不经意间被该模型误判。同时， LSTM 类 DGA 检测本身对非随机字符的字典类 DGA 的表现普遍不好，在此就不延伸讨论，请有兴趣的小伙伴自行探索。

## 无用功的原因

本文作者在 “A Death Match of Domain Generation Algorithms” 这片博客文章里讲解了伪随机类 DGA 的生成流程，其一般方法可以认为是基于日期的随机数种子加上若干移位操作生成一系列字符串。LSTM 或者更广义的 RNN 网络结构并不能学习到移位操作。它们能做的只是在固定数据集上利用字符的 n-gram 组合**拟合**出 DGA 算法的结果。然而这一拟合不能保证在新的数据集上也有同样正确的结果，相反，一些不正确的 n-gram 组合可能因为在这个数据集上碰巧有效从而被 LSTM 的反馈机制提到了更重要的位置，可是 LSTM 本身不能判别哪些组合实际有效哪些是碰巧。简单来说，它不能保证学习到的特征是判别 DGA 的有效特征，它只是通过**最少的努力**来学习到由训练数据集引入的对 DGA 字符特征的**偏见**。

这个问题不只是 LSTM 在 DGA 检测上存在，它更广泛的存在与其他深度学习问题，“Shortcut Learning in Deep Neural Networks” 这篇文章讨论了各种深度学习模型学习到的其实是**捷径**而不是真实有效的特征组合，就好比一个每次小考都可以作弊的“学霸”，在国家级大考这样监考严格的情况下，该“学霸”还能考出来好成绩么？

## 在线检测 DGA 该怎么做

既然把 LSTM 批判的这么惨，是不是基于字符集特征的在线 DGA 检测就完全没希望呢？既然市场有需要，还是可以做的。

### 基于字符集特征的办法

基于统计特征的算法虽然也容易收到样本攻击，但其手工特征工程的可解释性远强于 LSTM，即使有误报等情况，其可解释的特征也会为安全运营时的分流（Triaging）提供决策依据。学术界也有利用对抗样本训练可靠性更高的 LSTM 模型的办法，其目的是通过减少数据集引入的归纳偏见的办法提高 LSTM 的茁壮性，不过它需要更多的投入与测试，并且也需要定期部署更新模型。

### 基于 DNS 流量特征的办法

DGA 域名不只是有字符串特征，我们可以在 DGA 的离线算法里寻找机会。它们利用了更为通用和稳定的 DGA 域名在 DNS 流量里的序列或者共现特征，而非使用并不可靠的字符分布特征。DGA 域名一般在被感染主机生成后以一定序列发送，它在 DNS 流量日志里会混合于其他合法域名以序列形式存在，一个好的序列关联算法可以有效剔除噪声并找到有强关联的域名族群，经过进一步标定即可检测 DGA。这种方法除了对已知 DGA 有很好的检测，对未知 DGA 也很有效，但需要对当前 DNS 流量日志建模而非预训练部署。这类方法大致分为两类，基于序列特征和基于共现特征。

序列特征检测 DGA 的模型一般基于 word2vec 以及相关模型。本文作者在 2015 年的提交的专利 “System for correlation of domain names”里详细描述了这一办法，基本思路为对 DNS 查询序列取滑动窗口并对其中出现的域名对训练学习每个域名的向量表示，利用这些向量表示做聚类，并在聚类中寻找符合 DGA 网络行为特征（比如多数为 NXDOMAIN）的聚类。当然，这一方法并非本文作者独创，它在最近几年也被重复发现了多次，比如但不限于 “Vector representation of internet domain names using a word embedding technique” （2017） “Dns2Vec: Exploring Internet Domain Names Through Deep Learning” （2019）以及国内的文章 “Domain-Embeddings Based DGA Detection with Incremental Training Method” （2020），各位可以任选一篇对算法深入探索，在此无须赘述。这类办法投入到工业系统里的难点在于对于大规模数据的准实时化实现，并且需要高效可靠的模型训练机制，这两点在学术界探索的并不多。

共现特征检测 DGA 和序列特征思路相似，但是它避开了训练一个类 word2vec 模型的麻烦。在 360netlab 的 “A DGA Odyssey PDNS Driven DGA Analysis”中，它们利用域名 A 与 B 在该时间段内共同出现并解析到同一 IP 的特征矩阵，以此特征矩阵与自身转置想乘，再利用 Louvain 算法进行分割到若干聚类，并利用 DGA 网络行为进行进一步标定。聪明的小伙伴应该发现了，该算法其实并不需要十分昂贵的模型计算和日志存储，仅需要在一个时间窗口内积累到足够 louvain 进行分割的 DNS 流量即可，加以合理的工程投入，它可以作为可靠的准在线检测办法。

最后，有一位经验丰富的聪明小伙伴问了，有的 DGA 一次只生成一两个域名，那怎么通过聚类检测它们呢？类似于 `DBSCAN` 的算法也有 `cluster ID = -1` 这些与其他域名都聚不了类的嘛，和谁都不合群的域名也可能是有问题的。

## 总结

在使用数据模型解决安全类问题的时候，作者建议思考该模型是不是真的适合这一场景，而不是盲目认为其在当前数据集里拟合出的神奇效果足以支持它部署到生产系统里。本文也通过 LSTM 检测 DGA 的例子简单介绍了基于样本的模型攻击办法，以及模型的“捷径学习”，供各位小伙伴延伸阅读。

## 参考与延伸阅读

* CrowdStrike slides <https://www.slideshare.net/CrowdStrike/endtoend-analysis-of-a-domain-generating>
* CrowdStrike whitepaper <https://paper.bobylive.com/Meeting_Papers/BlackHat/USA-2013/US-13-Geffner-End-To-End-Analysis-of-a-Domain-Generating-Algorithm-Malware-Family-WP.pdf>
* 作者在乌云知识库的文章存档 <https://zhuanlan.zhihu.com/p/20045198>
* 作者开源的 SVM 检测 DGA <https://github.com/phunterlau/dga_classifier>
* Endgame paper <https://arxiv.org/abs/1611.00791> github <https://github.com/endgameinc/dga_predict>
* MaskDGA paper <https://arxiv.org/abs/1902.08909> github <https://github.com/liorsidi/Adversarial-DGA-Datasets>
* A Death Match of Domain Generation Algorithms <https://blogs.akamai.com/2018/01/a-death-match-of-domain-generation-algorithms.html>
* Shortcut Learning in Deep Neural Networks <https://arxiv.org/abs/2004.07780>
* System for correlation of domain names，专利编号 US 14/937616
* Vector representation of internet domain names using a word embedding technique <https://ieeexplore.ieee.org/document/8226415>
* Dns2Vec: Exploring Internet Domain Names Through Deep Learning <https://www.usenix.org/sites/default/files/conference/protected-files/scainet19_slides_arora.pdf>
* Domain-Embeddings Based DGA Detection with Incremental Training Method <https://arxiv.org/pdf/2009.09959.pdf>
* A DGA Odyssey PDNS Driven DGA Analysis <https://pc.nanog.org/static/published/meetings/NANOG71/1444/20171004_Gong_A_Dga_Odyssey__v1.pdf>