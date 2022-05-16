import copyreg
import random
import re
from sklearn.model_selection import KFold
import numpy
import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from tqdm import tqdm
from ..common.tools import save_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar
from nltk import word_tokenize
class TaskData(object):
    def __init__(self):
        pass
    def train_val_split(self,X, y,valid_size,stratify=False,shuffle=True,save = True,
                        seed = None,data_name = None,data_dir = None):
        pbar = ProgressBar(n_total=len(X))
        logger.info('split raw data into train and valid')
        if stratify:
            num_classes = len(list(set(y)))
            train, valid = [], []
            bucket = [[] for _ in range(num_classes)]
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                bucket[int(data_y)].append((data_x, data_y))
                pbar.batch_step(step=step,info = {},bar_type='bucket')
            del X, y
            for bt in tqdm(bucket, desc='split'):
                N = len(bt)
                if N == 0:
                    continue
                valid.extend(bt[N*valid_size//10:N*(valid_size+1)//10])
                train.extend(bt[:N*valid_size//10])
                train.extend(bt[N*(valid_size+1)//10:])


        train = self.sample(train, data_name)
        print("train",len(train),"vali",len(valid))
        if shuffle:
            random.seed(seed)
            random.shuffle(train)
        if save:
            train_csv = data_dir / f"{data_name}.train.csv"
            valid_csv = data_dir / f"{data_name}.valid.csv"
            csv_train = {"sentence": [], "label":[]}
            for i in train:
                csv_train["sentence"].append(i[0])
                csv_train["label"].append(i[1])
            t_csv = pd.DataFrame(csv_train)
            t_csv.to_csv(train_csv, index=False)
            csv_valid = {"sentence": [], "label":[]}
            for i in valid:
                csv_valid["sentence"].append(i[0])
                csv_valid["label"].append(i[1])
            v_csv = pd.DataFrame(csv_valid)
            v_csv.to_csv(valid_csv, index=False)
        return train, valid


    # data sample
    def sample(self,train:list, resource="dialog"):
        print(len(train))
        ob = []
        eb = []
        sr = []
        other = []
        for i in train:
            if i[1] == 0:
                ob.append(i)
                # new_data = self.data_augmentation(i, number=1,type="other")
                # ob.extend(new_data)
            elif i[1] == 1:
                eb.append(i)
                # new_data = self.data_augmentation(i, number=1)
                # eb.extend(new_data)
            elif i[1] == 2:
                sr.append(i)
                # new_data = self.data_augmentation(i, number=1)
                # sr.extend(new_data)
            else:
                # if random.randint(0,1) >= 0.5:
                other.append(i)
                # new_data = self.data_augmentation(i, number=1, type="Other")
                # other.extend(new_data)
        new_train = []
        new_train.extend(ob)
        new_train.extend(eb)
        new_train.extend(sr)
        new_train.extend(other)
        print(len(ob), len(eb), len(sr), len(other))
        return new_train

    def data_augmentation(self,sentence,number,type="label"):
        augment_list = []
        #
        if type == "label":
            aug = nac.OcrAug()
            text = aug.augment(sentence[0],n = number)
            if number == 1:
                augment_list.append((text, numpy.copy(sentence[1])))
            else:
                for a_text in text:
                    augment_list.append((a_text,numpy.copy(sentence[1])))

            aug = nac.KeyboardAug()
            text = aug.augment(sentence[0],n=number)
            if number == 1:
                augment_list.append((text, numpy.copy(sentence[1])))
            else:
                for a_text in text:
                    augment_list.append((a_text,numpy.copy(sentence[1])))

            aug = nac.RandomCharAug(action="insert")
            text = aug.augment(sentence[0],n=number)
            if number == 1:
                augment_list.append((text, numpy.copy(sentence[1])))
            else:
                for a_text in text:
                    augment_list.append((a_text,numpy.copy(sentence[1])))

            aug = naw.SpellingAug()
            text = aug.augment(sentence[0],n=number)
            if number == 1:
                augment_list.append((text, numpy.copy(sentence[1])))
            else:
                for a_text in text:
                    augment_list.append((a_text,numpy.copy(sentence[1])))

        aug = naw.RandomWordAug(action="swap")
        text = aug.augment(sentence[0],n=number)
        if number == 1:
            augment_list.append((text, numpy.copy(sentence[1])))
        else:
            for a_text in text:
                augment_list.append((a_text,numpy.copy(sentence[1])))

        return augment_list


    def read_data(self,raw_data_path,preprocessor = None,is_train=True,label2id=None):
        '''
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        targets,sentences = [],[]
        project = []
        fr = pd.read_csv(raw_data_path)
        for line in fr.itertuples():
            # if line[4] == 0:
            #     continue
            if is_train:
                # print(label2id)
                target = label2id[str(line[2])]
                sentence = str(line[1]).strip()
                if not ("_url_" in sentence or "_code_" in sentence or "_issue_" in sentence):
                    s = word_tokenize(sentence)
                    if len(s) <= 5:
                        continue


            else:
                lines = line.strip(',')
                target = -1
                sentence = str(lines).strip()

            if preprocessor:
                sentence = preprocessor(sentence)
            if sentence:
                # using for bee data
                # if sentence.startswith("at org."):
                #     continue
                targets.append(target)
                sentences.append(sentence)
                project.append(line[3])
        return targets,sentences, project

if __name__ == "__main__":
	pass
