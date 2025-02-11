import idx2numpy
import numpy as np
from torch.utils.data import Dataset
import torch
from pyvi import ViTokenizer
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MNISTDataset(Dataset):
    def __init__(self, image_path, label_path):
        super().__init__()
        
        # Đọc file hình ảnh và nhãn
        images = idx2numpy.convert_from_file(image_path).tolist()
        labels = idx2numpy.convert_from_file(label_path).tolist()
        
        # Lưu dữ liệu dưới dạng từ điển
        self.__data = {}
        for i_th, (image, label) in enumerate(zip(images, labels)):
            self.__data[i_th] = {
                'image': image,
                'label': label
            }

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        # Lấy ra hình ảnh và nhãn
        item = self.__data[idx]
        image = torch.tensor(item['image'], dtype=torch.float32)  # Chuyển đổi hình ảnh thành tensor kiểu float32
        label = torch.tensor(item['label'], dtype=torch.long)     # Chuyển đổi nhãn thành tensor kiểu long
        
        # Trả về dictionary chứa cả hai tensor
        return {
            'image': image,
            'label': label
        }



class UITVSFC(Dataset):
    def __init__(self, sents_path, sentiments_path, max_len = 100, pad_token='<PAD>', unk_token='<UNK>'):
        super().__init__()
        self.sents = pd.read_csv(sents_path, sep='.', header=None, index_col=None)[0]
        self.sentiments = pd.read_csv(sentiments_path, header=None, index_col=None)
        self.max_len = max_len
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sentiments = self.sentiments.values.flatten()
        self.word2idx = self.build_vocab()
        
    
        
    def build_vocab(self):
        X_train = pd.read_csv(r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\deeplearning\UIT-VSFC\train\sents.txt", sep='.', header=None, index_col=None)[0]
        print(X_train)
        V=[]
        for t in X_train:
            tokenized_sentence = ViTokenizer.tokenize(t)
            V = V + tokenized_sentence.split()
        V = list(set(V))
        m = {w : (i+2) for i, w in enumerate(V)}
        m[self.pad_token] = 0
        m[self.unk_token] = 1
        return m
    
    def vocab_size(self):
        return len(self.word2idx)
    
    def __len__(self):
        return len(self.sents)
    
    def encodeSen(self, text):
        tokenized_sentence = ViTokenizer.tokenize(text)
        list_w =  [self.word2idx.get(w, self.word2idx['<UNK>']) for w in tokenized_sentence.split()]
        return pad_sequences([list_w], maxlen=self.max_len, padding='post', value = self.word2idx[self.pad_token])[0]

    def __getitem__(self, idx):
        sent = self.sents[idx]
        sentiment = self.sentiments[idx]
        sent = self.encodeSen(sent)
        return torch.tensor(sent, dtype=torch.long), torch.tensor(sentiment, dtype=torch.long)

# train_dataset = UITVSFC(
#     sents_path=r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\deeplearning\UIT-VSFC\train\sents.txt",
#     sentiments_path=r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\deeplearning\UIT-VSFC\train\sentiments.txt"
# )

import json
class UITOCD(Dataset):
    def __init__(self, data_path, max_len = 100, pad_token='<PAD>', unk_token='<UNK>'):
        super().__init__()
        self.data = self.load_data(data_path)
        self.review = [x['review'] for x in self.data.values()]
        self.label = [x['label'] for x in self.data.values()]
        self.max_len = max_len
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = self.build_vocab()
        
    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    def build_vocab(self):
        # X_train = json.load(open(r"C:\Users\VIET HOANG - VTS\Downloads\UIT-ViOCD\UIT-ViOCD\train.json"), encoding='utf-8')
        with open(r"C:\Users\VIET HOANG - VTS\Downloads\UIT-ViOCD\UIT-ViOCD\train.json", 'r', encoding='utf-8') as file:
            X_train = json.load(file)
        X_train = [x['review'] for x in X_train.values()]

        V=[]
        for t in X_train:
            tokenized_sentence = ViTokenizer.tokenize(t)
            V = V + tokenized_sentence.split()
        V = list(set(V))
        m = {w : (i+2) for i, w in enumerate(V)}
        m[self.pad_token] = 0
        m[self.unk_token] = 1
        return m
    
    def vocab_size(self):
        return len(self.word2idx)
    
    def __len__(self):
        return len(self.review)
    
    def encodeRev(self, text):
        tokenized_sentence = ViTokenizer.tokenize(text)
        list_w =  [self.word2idx.get(w, self.word2idx['<UNK>']) for w in tokenized_sentence.split()]
        return pad_sequences([list_w], maxlen=self.max_len, padding='post', value = self.word2idx[self.pad_token])[0]

    def encodeLabel(self, label):
        dic = {
            "non-complaint" : 0,
            "complaint" : 1
        }

        return dic[label]

    def __getitem__(self, idx):
        review = self.review[idx]
        label = self.label[idx]
        review = self.encodeRev(review)
        label = self.encodeLabel(label)
        return torch.tensor(review, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    

