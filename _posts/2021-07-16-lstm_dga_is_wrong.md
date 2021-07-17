# Why using LSTM to detect DGA is a wrong idea

Disclaimer: Nothing in this blog is related to the author's day-to-day work. The content is neither affiliated nor sponsored by any companies.

Link to the Chinese version <https://toooold.com/2021/07/12/dga_detection.html>

The short answer is that, the char feature based DGA detection model with LSTM is vulnerable to adversarial attack, and its 99.9% benchmark score on a fixed dataset can lead to significant risk when deployed to the real-world DNS traffic at millions of QPS. The reason behind LSTM when solving this problem comes from shortcut learning. In this article, we briefly introduce sample-based adversarial and some "shortcut learning" analysis, starting from the investigation of "why LSTM detects DGA is wrong". Papers and open source code mentioned here are listed in the reference section for your reference.

## Char feature based DGA detection

The detection of domain name generation algorithm (DGA) botnets from DNS traffic logs has been a hot topic, and the methods are generally divided into two categories: classification models using domain name character features and clustering models using DNS traffic sequence and cooccurrence features, with the first being more popular because it is easier to deploy online models and also requires less resources, which is more appropriate for scenarios in which DNS logs cannot be easily collected from a security service box.

Character feature-based classification models use string patterns from DGA domains to train and produce binary or multi-classification models, such as randomness of frequency and position, vowel-consonant mixture ratio, n-gram combinations, and so on, and typically use known DGAs as positive samples and Alexa Top domains as negative samples. CrowdStrike's paper "End-to-End Analysis of a Domain Generating Algorithm Malware Family" in BlackHat 2013 is one of the earliest examples in product. I published an article in 2015 titled "Identifying Randomly Generated C&C Domains with Machine Learning" in which I used SVM to detect DGA in Wooyun and open sourced the code; the security industry recognized FANCI in usenixsecurity18 published "FANCI : Feature-based Automated NXDomain Classification and Intelligence" and related open source code in 2018. These classification models incorporate statistical-based feature engineering with varying degrees of inventiveness from different authors, and many of them achieve more than 99% accuracy in their respective datasets.

Because we are now living in the deep learning era, why not use a deep learning model to learn these features?

### LSTM looks more "deep learning"

In late 2016, Endgame (now Elastic) researchers developed a simple LSTM that uses the order of occurrence of adjacent characters in a domain name as input to build a classification model; in 2018, the article Duc Tran et al "A LSTM-based framework for handling multiclass imbalance in DGA botnet detection" presents an improvement on unbalanced classification, as well as the open source code LSTM. MI has been mentioned numerous times in the industry. Both are based on this simple LSTM network structure (from endgame's open-source code and LSTM. MI makes use of the same).

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

Those who are familiar with LSTM can tell that it is nearly identical to the LSTM examples in the text classification tutorials, even the length of the encoding vector 128 remains the same, and the features it learns can be considered as a subset of n-gram class features for this dataset. So,  are we able to attack and fool this model?

In fact, the effective attack method is much simpler than you believe, and these LSTM models are more vulnerable than you think.

### Simple attack to LSTM DGA detector

There are white-box attacks that know the training dataset or network structure and corresponding black-box approaches for LSTM DGA detection.

"CharBot: A Simple and Effective Method for Evading DGA Classifiers" is a white-box attack method that requires very little programming: choose a random domain name from the training dataset's negative samples and randomly pick and switch two chars into another two, then replace the TLD. This simple attack can reduce the accuracy of models like LSTM.MI to lower than 90%, and the detection rate can be as low as 30%.

Some black-box attack models, such as "MaskDGA: A Black-box Evasion Technique Against DGA Classifiers and Adversarial Defenses," are also interesting. By simply creating a shallow adversarial network, one can reduce the F-1 score of models like LSTM-MI from 0.977 to 0.495, making it indistinguishable from a random guess. DeepDGA, DeceptionDGA, and other similar approaches take a similar approach.

As we can see, LSTM DGA detection is extremely vulnerable to simple sample-based adversarial attacks, not to mention that the model's claimed 99.9 percent accuracy may be incorrect for domains appearing in large-scale DNS traffic logs outside the validation dataset. LSTM DGA detection, on the other hand, generally performs poorly for dictionary-like DGAs with non-random characters like `suppobox`.

## The reason behind it

In my previous blog post "A Death Match of Domain Generation Algorithms", I have described the process of generating a pseudo-random type of DGAs, whose general approach can be thought of as a series of strings generated by adding a number of shift operations to a date-based random number seed. Such "shift" operations are not learned by LSTM or, more broadly, RNN network structures. They can only **fit** the DGA algorithm results to a fixed data set using an n-gram combination of characters. This fit, however, does not guarantee the same correct results on a new dataset; instead, some incorrect n-gram combinations may happen to be valid on this dataset and thus are given more weight by the LSTM's propagation mechanism, but LSTM itself cannot distinguish which combinations are actually valid and which ones just happen to be. Simply put, it cannot guarantee that the learned features are valid for distinguishing DGAs; it only learns the **bias** introduced by the training dataset for DGA character features with **minimal effort**.

This issue exists not only for LSTM on DGA detection, but also for other deep learning problems. The article "Shortcut Learning in Deep Neural Networks" discusses how different deep learning models learn **shortcuts** rather than true and valid feature combinations. Can we trust a person who always take a shortcut? It is the same with machine learning.

## So, any hope for an online algorithm

Is online DGA detection based on character set features completely hopeless now that LSTM has been heavily criticized? It can still be done since there is a market need.

### The char feature method

While statistical feature-based algorithms are vulnerable to sample-based adversarial attacks as well, their handcrafted feature engineering is far more interpretable than LSTMs, and their interpretable features provide a foundation for decisions on triaging during security operations, even in cases of false positives. There are also additional academic approaches to training more reliable LSTM models using adversarial samples, which aim to improve LSTM robustness by reducing induction bias introduced by the dataset, but it requires more frequent update of input and testing, as well as regular deployment of updated models.

### The DNS traffic log method

DGA domains are more than just strings, and we can look for opportunities in the offline detection algorithms. Instead of relying on untrustworthy character distribution features, they use more generic and stable sequence or co-occurrence features of DGA domains in DNS traffic. This method detects both known and unknown DGAs, but it requires modeling of current DNS traffic logs rather than pre-trained deployments. These methods are divided into two types: sequence-based features and co-occurrence-based features.

Word2vec and related models are commonly used in DGA models for sequential feature detection. This approach is detailed in my patent "System for domain name correlation" submitted in 2015. The basic idea is to use a sliding window of DNS query sequences to learn vector representations of each domain name by training pairs of domain names that appear in them, and then use these vector representations to cluster. These vector representations are used to form clusters, and within the clusters, we find clusters that match the DGA network's behavioral characteristics (e.g., mostly NXDOMAIN). This approach, of course, is not unique to my idea, and it has been re-discovered several times in recent years, including but not limited to "Vector representation of internet domain names using a word embedding technique" (2017), "Dns2Vec: Exploring Internet Domain Names Through Deep Learning" (2019), and " Domain-Embeddings Based DGA Detection with Incremental Training Method" (2020). One can choose any one of them to explore further about the algorithm and the idea behind it, so there is no need to elaborate here. The true difficulty in implementing such approaches in industrial systems stems from the need for near-real-time implementation of large-scale data as well as efficient and reliable model training mechanisms, both of which have received little attention in academia.

Detection of cooccurrence features DGA is similar to the idea of sequence features, but it bypasses the hassle of training a word2vec-like model, which is expensive. In [360netlab](https://twitter.com/360netlab)'s "A DGA Odyssey PDNS Driven DGA Analysis," they use the feature matrix of domains A and B that co-occur in that time period and resolve to the same IP, multiply this feature matrix with its own transpose, and then use Louvain's algorithm to segment into clusters, and use DGA network behavior for further calibration. I believe the reader should realize that this algorithm does not need a very expensive model computation or log storage, but only needs to accumulate enough DNS traffic for Louvain clustering within a time window. With reasonable engineering investment, it can be a reliable near-online detection method.

Finally, people may ask, since some DGAs only generate one or two domains at a time, how can clustering detect them in this method? `DBSCAN` and other cluster algorithms can bring in `cluster ID = -1`: if domains are not clustered with other domains, not fit in with anyone, they may be suspicious.

## Summary

When using a data model to solve a security-related problem, I suggest considering whether the model is truly appropriate for the scenario, rather than assuming that the magical silver bullet it fits in the current dataset is sufficient to support its deployment in a production system. This article briefly introduces the sample-based model attack approach and model "shortcut learning" using the LSTM detection DGA example. Please explore further with the reference list.

## Reference and further reading

* CrowdStrike slides <https://www.slideshare.net/CrowdStrike/endtoend-analysis-of-a-domain-generating>
* CrowdStrike whitepaper <https://paper.bobylive.com/Meeting_Papers/BlackHat/USA-2013/US-13-Geffner-End-To-End-Analysis-of-a-Domain-Generating-Algorithm-Malware-Family-WP.pdf>
* "Identifying Randomly Generated C&C Domains with Machine Learning" in Chinese <https://zhuanlan.zhihu.com/p/20045198>
* My open sourced DGA detector <https://github.com/phunterlau/dga_classifier>
* Endgame paper <https://arxiv.org/abs/1611.00791> github <https://github.com/endgameinc/dga_predict>
* MaskDGA paper <https://arxiv.org/abs/1902.08909> github <https://github.com/liorsidi/Adversarial-DGA-Datasets>
* A Death Match of Domain Generation Algorithms <https://blogs.akamai.com/2018/01/a-death-match-of-domain-generation-algorithms.html>
* Shortcut Learning in Deep Neural Networks <https://arxiv.org/abs/2004.07780>
* My patent "System for correlation of domain names", US 14/937616
* Vector representation of internet domain names using a word embedding technique <https://ieeexplore.ieee.org/document/8226415>
* Dns2Vec: Exploring Internet Domain Names Through Deep Learning <https://www.usenix.org/sites/default/files/conference/protected-files/scainet19_slides_arora.pdf>
* Domain-Embeddings Based DGA Detection with Incremental Training Method <https://arxiv.org/pdf/2009.09959.pdf>
* 360netlab: A DGA Odyssey PDNS Driven DGA Analysis <https://pc.nanog.org/static/published/meetings/NANOG71/1444/20171004_Gong_A_Dga_Odyssey__v1.pdf>