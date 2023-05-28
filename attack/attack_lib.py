
from nltk.tokenize import RegexpTokenizer
from english import ENGLISH_FILTER_WORDS
import random
import sys
from word2vec.word2vec_embed import Word2VecSubstitute
import shutil
import numpy as np
from compute_img_sim import compute_img_sim

sys.path.append('../target_model/min_dalle')
from image_from_text import dalle_mini_gen_img_from_text

class Error(Exception):
    """Base class for other exceptions"""
    pass

class WordNotInDictionaryException(Error):
    """Raised when the input value is too small"""
    pass

class attack():
    def __init__(self, tar_sent):
        #nothing to be initialized
        
        self.count = 0

        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(tar_sent.lower())

        #filter unimportant word

        target_word_ls = []
        for token in tokens:
            if token.lower() in ENGLISH_FILTER_WORDS:
                continue
            target_word_ls.append(token)
        self.target_sent_tokens = target_word_ls
        print("tar_sent_tokens: ", self.target_sent_tokens)

        self.Word2vec = Word2VecSubstitute(tar_tokens=self.target_sent_tokens)

        print("initialize attack class.")

    
    def selectBug(self, original_word, if_initial=False, word_idx=None, x_prime=None):
        bugs = self.generateBugs(original_word, if_initial)
        target_num = random.randint(0, len(bugs)-1)
        bugs_ls = list(bugs.values())
        #randomly select a bug to return
        bug_choice = bugs_ls[target_num]
        return bug_choice
    

    def replaceWithBug(self, x_prime, word_idx, bug):
        return x_prime[:word_idx] + [bug] + x_prime[word_idx + 1:]

    def generateBugs(self, word, if_initial=False, sub_w_enabled=False, typo_enabled=False):
        
        if if_initial:
            bugs = {"insert": word, "sub_W": word, "del_C": word, "sub_tar_W": word}
            if len(word) <= 2:
                return bugs
            bugs["insert"] = self.bug_insert(word)
            bugs["sub_W"] = self.bug_sub_W(word)
            bugs["del_C"] = self.bug_delete(word)
            bugs["sub_tar_W"] = self.bug_sub_tar_W(word)
        else:
            bugs = {"sub_W": word, "ins_C": word}
            if len(word) <= 2:
                return bugs
            bugs["sub_W"] = self.bug_sub_W(word)
            bugs["ins_C"] = self.bug_insert(word)

        return bugs

    def bug_sub_tar_W(self, word):
        word_index = random.randint(0, len(self.target_sent_tokens) - 1)
        tar_word = self.target_sent_tokens[word_index]
        res = self.Word2vec.substitute(tar_word)
        if len(res) == 0:
            return word
        return res[0][0]

    def bug_sub_W(self, word):
        try:
            res = self.Word2vec.substitute(word)
            if len(res) == 0:
                return word
            return res[0][0]
        except WordNotInDictionaryException:
            return word

    def bug_insert(self, word):
        if len(word) >= 6:
            return word
        res = word
        point = random.randint(1, len(word) - 1)
        #insert _ instread " "
        res = res[0:point] + "_" + res[point:]
        return res

    def bug_delete(self, word):
        res = word
        point = random.randint(1, len(word) - 2)
        res = res[0:point] + res[point + 1:]
        return res

    def bug_swap(self, word):
        if len(word) <= 4:
            return word
        res = word
        points = random.sample(range(1, len(word) - 1), 2)
        a = points[0]
        b = points[1]

        res = list(res)
        w = res[a]
        res[a] = res[b]
        res[b] = w
        res = ''.join(res)
        return res

    def bug_random_sub(self, word):
        res = word
        point = random.randint(0, len(word)-1)

        choices = "qwertyuiopasdfghjklzxcvbnm"
        
        subbed_choice = choices[random.randint(0, len(list(choices))-1)]
        res = list(res)
        res[point] = subbed_choice
        res = ''.join(res)
        return res
    
    def bug_convert_to_leet(self, word):
        # Dictionary that maps each letter to its leet speak equivalent.
        leet_dict = {
            'a': '4',
            'b': '8',
            'e': '3',
            'g': '6',
            'l': '1',
            'o': '0',
            's': '5',
            't': '7'
        }
        
        # Replace each letter in the text with its leet speak equivalent.
        res = ''.join(leet_dict.get(c.lower(), c) for c in word)
        
        return res


    def bug_sub_C(self, word):
        res = word
        key_neighbors = self.get_key_neighbors()
        point = random.randint(0, len(word) - 1)

        if word[point] not in key_neighbors:
            return word
        choices = key_neighbors[word[point]]
        subbed_choice = choices[random.randint(0, len(choices) - 1)]
        res = list(res)
        res[point] = subbed_choice
        res = ''.join(res)

        return res

    def get_key_neighbors(self):
        ## TODO: support other language here
        # By keyboard proximity
        neighbors = {
            "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
            "i": "uojkl", "o": "ipkl", "p": "ol",
            "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
            "j": "yuihknm", "k": "uiojlm", "l": "opk",
            "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
        }
        # By visual proximity
        neighbors['i'] += '1'
        neighbors['l'] += '1'
        neighbors['z'] += '2'
        neighbors['e'] += '3'
        neighbors['a'] += '4'
        neighbors['s'] += '5'
        neighbors['g'] += '6'
        neighbors['b'] += '8'
        neighbors['g'] += '9'
        neighbors['q'] += '9'
        neighbors['o'] += '0'

        return neighbors

def sort_words_by_importance(tokenizer, save_dir, ori_sent, tar_img_path):
    tokens = tokenizer.tokenize(ori_sent.lower())
    sim_ls = []

    for i in range(len(tokens)):
        new_tokens = tokens[:i] + tokens[i+1:]
        x_prime_sent = " ".join(new_tokens)
        
        x_img_path = save_dir + "gen.png"

        dalle_mini_gen_img_from_text(x_prime_sent, x_img_path)

        similarity = compute_img_sim(x_img_path, tar_img_path)
        sim_ls.append(similarity.item())

    sim_arr = np.array(sim_ls)   
    scores_logits = np.exp(sim_arr - sim_arr.max()) 
    sim_probs = scores_logits / scores_logits.sum()
    
    return sim_probs
