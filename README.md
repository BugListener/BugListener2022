# BugListener

This is a replication package for `BugListener: Identifying and Synthesizing Bug Reports from Collaborative Live Chats`. 
Our project is public at: <https://github.com/BugListener/BugListener2022>

## Content

1. [Project Structure](#1-Project-Structure)<br>
2. [Project Summary](#2-Project-Summary)<br>
3. [Models](#3-Models)<br>
4. [Experiments](#4-Experiments)<br>
&ensp;&ensp;[4.1 Baselines](#41-Datasets)<br>
&ensp;&ensp;[4.2 Our Dataset](#42-Baselines)<br>
5. [Results](#5-Results)<br>
&ensp;&ensp;[5.1 RQ1: Bug Reports Identification](#51-rq1-Bug-Reports-Identification)<br>
&ensp;&ensp;[5.2 RQ2: Bug Reports Synthesis](#52-rq2-Bug-Reports-Synthesis)<br>
&ensp;&ensp;[5.3 RQ3: Component Analysis](#53-rq3-component-analysis)<br>
6. [Human Evaluation](#6-Human-Evaluation)<br>

## 1 Project Structure
- `data/`
	- `*_bug.json :bug-report dialogs data`
	- `*_other.json :non bug-report dialogs data`


- `dataloader.py : dataset reader for BugListener`
- `model.py : BugListener model`
- `FocalLoss.py : focal loss function`
- `train.py : a file for model training`
- `bert_classfier: sentence classfier for OB、EB、SR`

## 2 Project Summary
In community-based software development, developers frequently rely on live-chatting to discuss emergent bugs/errors they encounter in daily development tasks. However, it remains a challenging task to accurately record such knowledge due to the noisy nature of interleaved dialogs in live chat data. In this paper, we first formulate the task of identifying and synthesizing bug reports from community live chats, and propose a novel approach, named BugListener, to address the challenges. Specifically, BugListener automates three sub-tasks: 1) Disentangle live chat logs; 2) Identify the bug-report dialogs; 3) Synthesize the bug reports.

## 3 Models
The structure of BugListener is shown as follow:
![](https://github.com/BugListener/BugListener2022/blob/master/diagrams/approach.png)
I.	The *Dialog Disentanglement* first uses the pipeline of data preprocessing, i.e., spell checking, low-frequency token replacement, acronym and emoji replacement, and broken utterance removal. Then we choose the SOTA model [irc-disentanglement](https://github.com/jkkummerfeld/irc-disentanglement/zipball/master) to seperate the whole chat log into independent dialogs with reply-to relationships.

II.	The *Utterance Embedding Layer* aims to encode semantic information of words, as well as to learn the representation of utterances. We first encode each word in utterances into a semantic vector by utilizing the deep pre-trained BERT model. Then, we use TextCNN to learn the utterance representation.

III. The *Graph-based Context Embedding layer* aims to capture the graphical context of utterances in one dialog. we first construct a dialog graph. Then, we learn the contextual information of the dialog graph via a two-layer graph neural network. 

IV.	The *Dialog Embedding and Classification layer* aims to obtain the representation of an entire dialog and classify it as either a positive or a negative bug-report dialog. We First utilize the Sum-Pooling and the Max-Pooling layer to obtain the dialog embedding. Then, the label is predicted by feeding the dialog embedding into two Full-Connected (FC) layers followed by the Softmax function.

V.	In this layer, we synthesize the bug reports by utilizing the TextCNN model and Transfer Learning network to classify the sentences into three groups: observed behaviors (OB), expected behaviors (EB), and steps to reproduce the bug (SR).

## 4 Experiments
We propose three RQs in our paper, which is related to our experiment:
- RQ1: How effective is BugListener in identifying bug-report dialogs from live chat data?
- RQ2: How effective is BugListener in synthesizing bug reports?
- RQ3 How does each individual component in BugListener contribute to the overall performance?

### 4.1 Datasets
The statistics of our experiment dataset is as follows: 
![](https://github.com/BugListener/BugListener2022/blob/master/diagrams/dataset.png)
where Part, Dial, Uttr, Sen are short for participating developers, dialog, utterance, and sentence, respectively. BR and NBR denote bug-report and non-bug-report dialogs. 𝑈<sub>𝑟</sub> denotes sentences in reporter's utterances, and 𝑈<sub>𝑟</sub>' denotes the pruned 𝑈<sub>𝑟</sub>.
### 4.2 Baselines
For RQ1, we compare our BugListner with common baselines: i.e., Naive Bayesian (NB), Random Forest (RF), Gradient Boosting Decision Tree (GBDT), and FastText(FT); additional baselines: i.e., Casper, CNC, and DECA_PD.

For RQ2: we compare our BugListner with common baselines: i.e., Naive Bayesian (NB), Random Forest (RF), Gradient Boosting Decision Tree (GBDT), and FastText(FT); additional baselines: i.e.,BEE.

For RQ3, we compare BugListener with its two variants in bug report identification task: 1) BugListener w/o CNN, which removes the TextCNN. 2) BugListener w/o GNN, which removes the graph neural network. We compare BugListener with its variant without transfer learning (i.e., BugListener w/o TL) in bug report synthesis task.

## 5 Results
### 5.1 RQ1: (Bug Reports Identification)
The following table shows the comparison results between the performance of BugListener and those of the seven baselines across data from six OSS communities.
![](https://github.com/BugListener/BugListener2022/blob/master/diagrams/RQ1.png)

Answering RQ1: when comparing with the best Precision-performer among the seven baselines, i.e., GBDT, BugListener can improve its average precision by 5.66%. Similarly, BugListener improves the best Recall-performer, i.e., FastText, by 7.56% for average recall, and improves the best F1 performer, i.e., CNC, by 10.37% for average F1. At the individual project level, BugListener can achieve the best performance in most of the six communities.

### 5.2 RQ2: (Bug Reports Synthesis)
The following figure summarizes the comparison results between the average performance of BugListener and the five baselines.
<div align=center><img src="https://github.com/BugListener/BugListener2022/blob/master/diagrams/RQ2.png" width="550" alt="dd-test"/></div><br>
Answering RQ2: BugListener can achieve the highest performance in predicting OB, EB, and SR sentences. It outperforms the six baselines in terms of F1. For predicting OB sentences, it reaches the highest F1 (67.37%), improving the best baseline GBDT by 7.21%. For predicting EB sentences, it reaches the highest F1 (87.14%), improving the best baseline FastText by 7.38%. For predicting SR sentences, it reaches the highest F1 (65.03%), improving the best baseline FastText by 5.30%.

### 5.3 RQ3: (Component Analysis)
The figure (a) presents the performances of BugListener and its two vari-ants for BRI task. The figure (b) shows the performance of BugListener and its variant without transfer technique for BRS task.
<div align=center><img src="https://github.com/BugListener/BugListener2022/blob/master/diagrams/RQ3.png" width="550" alt="dd-test"/></div><br>
Answering RQ3: For BRI task: When compared with BugListener and BugListener w/o GNN, removing the GNN component will lead to a dramatic decrease of the average F1 (by 9.87%) across all the communities. When compared with BugListener and BugListener w/o CNN, removing the TextCNN component will lead to the average F1 declines by 8.21%. For BRS task. We can see that, without the transfer learning from large external bug reports dataset, the F1 will averagely decrease by 3.26%, 6.45%, 14.90% for OB, EB, and SR prediction, respectively.

## 6 Human Evaluation
To further demonstrate the generalization and usefulness of our approach, we apply BugListener on recent live chats from five new communities: Webdriverio, Scala, Materialize, Webpack, and Pandas (note that these are different from our studied communities so that all data of these communities do not appear in our training/testing data). Then we ask human evaluators to assess the correctness, quality, and usefulness of the bug reports generated by BugListener.  
The complete survey containing 31 bug reports can be downloaded with [Link](https://github.com/BugListener/BugListener2022/blob/master/data/human%20evaluation.xlsx).

**Procedure**. First, we crawl the recent one-month (July 2021 to August 2021) live chats of the five new communities from Gitter, which contain 3,443 utterances. Second, we apply BugListener to disentangle and construct the live chats into about 562 separated dialogs. Among them, BugListener identifies 31 potential bug reports in total. We recruit nine developers with experience in using or contributing to the five open source communities. For each participant, we assign 9-11 bug reports of the communities that they are familiar with. Each bug report is evaluated by three participants. For each bug report, each participant has the following information available: (1) the associated open source community; (2) the original textual dialogs from Gitter; (3) the bug report generated by BugListener.  
The survey contains three questions: (1) Correctness: Whether the dialog is discussing a bug that should be reported at that moment (Yes or No)? (2) Quality: How would you rate the quality of Description, Observed Behavior, Expected Behavior, and Step to Reproduce in the bug report (using a five-level Likert scale)? (3) Usefulness: How would you rate the usefulness of BugListener (using a 5-level Likert scale)?

For each dialog, the ground truth is obtained based on the majority vote from the three participants, and we use the average score of the three evaluations as the final score. 
<div align=center><img src="https://github.com/BugListener/BugListener2022/blob/master/diagrams/HE.png" width="550" alt="dd-test"/></div><br>
Fig (a) shows the bar and pie chart depicting the correctness of BugListener. Among the 31 bug reports identified by BugListener, 24 (77%) of them are correct, while 7 (23%) of them are incorrect. The bar chart shows the correctness distributed among the five communities. The correctness ranges from 63% to 100%. The perceived correctness indicates that BugListener is likely generalized to other open source communities with a relatively good and stable performance. 

Fig (b) shows an asymmetric stacked bar chart depicting the perceived quality and usefulness of BugListener’s bug reports, in terms of description, observed behavior, expected behavior, and step to reproduce. We can see that, the high quality of bug report description is highly admitted, 85% of the responses agree that the bug report description is satisfactory (i.e., “somewhat satisfied” or “satisfied”). The high quality of OB, EB, and S2R are also moderately admitted (62%, 46%, and 58% on aggregated cases, respectively). In addition, the usefulness bar chart shows that 71% of participants agree that BugListener is useful.
