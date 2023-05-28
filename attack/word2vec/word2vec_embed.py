from typing import Dict, Optional
import torch
import numpy as np
import pickle
import os
import random 
import re
import nltk


class Error(Exception):
    """Base class for other exceptions"""
    pass


class WordNotInDictionaryException(Error):
    """Raised when the input value is too small"""
    pass



class WordEmbedding():
    def __init__(self, word2id : Dict[str, int], embedding) -> None:
        self.word2id = word2id
        self.embedding = embedding
    
    def transform(self, word, token_unk):
        if word in self.word2id:
            return self.embedding[ self.word2id[word] ]
        else:
            if isinstance(token_unk, int):
                return self.embedding[ token_unk ]
            else:
                return self.embedding[ self.word2id[ token_unk ] ]



def LOAD(path):
    word2id = pickle.load( open( os.path.join(path, "word2id.pkl"), "rb") )
    wordvec = pickle.load( open( os.path.join(path, "wordvec.pkl"), "rb") )
    return WordEmbedding(word2id, wordvec)


def LOAD_perceptron_tagger(path):
    ret = __import__("nltk").tag.PerceptronTagger(load=False)
    ret.load("file:" + os.path.join(path, "averaged_perceptron_tagger.pickle"))
    return ret.tag


_POS_MAPPING = {
    "JJ": "adj",
    "VB": "verb",
    "NN": "noun",
    "RB": "adv"
}

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def check_word_in_tokens(word, tokens):
    flag = False
    for token in tokens:
        if token == word:
            flag = True
            break
    return flag

class Word2VecSubstitute():
    
    def __init__(self, tar_tokens=None, cosine=False, k = 100, threshold = 10, device = None):
        """
        Embedding based word substitute.

        Args:
            word2id: A `dict` maps words to indexes.
            embedding: A word embedding matrix.
            cosine: If `true` then the cosine distance is used, otherwise the Euclidian distance is used.
            threshold: Distance threshold. Default: 0.5
            k: Top-k results to return. If k is `None`, all results will be returned. Default: 50
            device: A pytocrh device for computing distances. Default: "cpu"
        
        """

        if device is None:
            device = "cpu"
        
        #load wordvec
        wordvec = LOAD("../Word2Vec/")

        self.tar_tokens = tar_tokens
        self.word2id = wordvec.word2id
        self.embedding = torch.from_numpy(wordvec.embedding)
        self.cosine = cosine
        self.k = k
        self.threshold = threshold

        self.id2word = {
            val: key for key, val in self.word2id.items()
        }
        
        if cosine:
            self.embedding = self.embedding / self.embedding.norm(dim=1, keepdim=True)
        
        self.embedding = self.embedding.to(device)
        self.pos_tagger = LOAD_perceptron_tagger("../NLTKPerceptronPosTagger/")


    def get_pos(self, word, pos_tagging=True):
        
        tokens = [word]
        for word, pos in self.pos_tagger(tokens):
            if pos[:2] in _POS_MAPPING:
                mapped_pos = _POS_MAPPING[pos[:2]]
            else:
                mapped_pos = "other"
        return mapped_pos


    def substitute(self, word):
        if word not in self.word2id:
            return []
        
        #get pos of word
        ori_pos = self.get_pos(word)
        
        wdid = self.word2id[word]
        wdvec = self.embedding[wdid, :]
        if self.cosine:
            dis = 1 - (wdvec * self.embedding).sum(dim=1)
        else:
            dis = (wdvec - self.embedding).norm(dim=1)

        idx = dis.argsort()

        if self.k is not None:
            idx = idx[:self.k]
        
        #filter index dis that are larger than threshold
        
        output_idx = []
        for i in idx:
            if dis[i] < self.threshold and dis[i] != 0:
                #print(f"idx: {i}, dis[i]: {dis[i]}")
                output_idx.append(i.item())
        
        idx = output_idx
        new_idx = []
        #filter strange long word
        for id_ in idx:
            flag = True
            pos = self.get_pos(self.id2word[id_])

            if "_" in self.id2word[id_]:
                word_slc = self.id2word[id_].split("_")
                
                for word_s in word_slc:
                    if isEnglish(word_s)==False:
                        flag = False 
                        break
            else:
                if isEnglish(self.id2word[id_])==False:
                    flag = False
            
            #check if in target token list
            is_in_target = False
            if self.tar_tokens is not None:
                is_in_target = check_word_in_tokens(self.id2word[id_], self.tar_tokens)

            if flag and pos == ori_pos:
                if not is_in_target:
                    new_idx.append(id_)      
        
        if len(new_idx) == 0:
            return []
        idx = random.choices(new_idx, k=1)

        return [
            (self.id2word[id_], dis[id_].item()) for id_ in idx
        ]