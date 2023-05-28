import numpy as np
import os
import sys
import random
import shutil
from nltk.tokenize import RegexpTokenizer
from english import ENGLISH_FILTER_WORDS
from compute_img_sim import compute_img_sim
from attack_lib import attack
from attack_lib import sort_words_by_importance
sys.path.append('../target_model/min_dalle')
from image_from_text import dalle_mini_gen_img_from_text
import argparse


def check_if_contains(tokens):
    flag = False
    loc = 0
    for token in tokens:
        if "_" in token:
            flag = True
            break
        loc += 1
    return flag, loc

def check_if_in_list(sent, sent_ls):
    flag = False
    for tar_sent in sent_ls:
        if sent == tar_sent:
            flag = True
            break
    return flag


def get_new_pop(elite_pop, elite_pop_scores, pop_size):
    scores_logits = np.exp(elite_pop_scores - elite_pop_scores.max()) 
    elite_pop_probs = scores_logits / scores_logits.sum()

    cand1 = [elite_pop[i] for i in np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    cand2 = [elite_pop[i] for i in np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]

    #exchange two parts randomly
    mask = np.random.rand(pop_size, len(elite_pop[0])) < 0.5 
    
    next_pop = []
    pop_index = 0
    for pop_flag in mask:
        pop = []
        word_index = 0
        for word_flag in pop_flag:
            if word_flag:
                pop.append(cand1[pop_index][word_index])
            else:
                pop.append(cand2[pop_index][word_index])
            word_index += 1
        next_pop.append(pop)
        pop_index += 1

    return next_pop
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

class Genetic():
    
    def __init__(self, ori_sent, tar_img_path, tar_sent, log_save_path, intem_img_path, best_img_path, mutate_by_impor):
        
        self.init_pop_size = 150
        self.pop_size = 15
        self.elite_size = 8
        self.mutation_p = 0.85
        self.mu = 0.99
        self.alpha = 0.001
        self.max_iters = 50
        self.store_thres = 80

        self.target_img_path = tar_img_path
        self.log_save_path = log_save_path
        self.intermediate_path = intem_img_path
        self.best_img_path = best_img_path
        self.target_sent = tar_sent
        self.mutate_by_impor = mutate_by_impor
        
        #initialize attack class
        self.attack_cls = attack(self.target_sent)
        
        #initialize tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')
        tokens = self.tokenizer.tokenize(ori_sent.lower())     

        #generate large initialization corpus
        self.pop = self.initial_mutate(tokens, self.init_pop_size)
        print("initial pop: ", self.pop)

    def initial_mutate(self, pop, nums):
        #random select the pop sentence that will mutate 
        new_pop = [pop]
        new_sent_ls = [" ".join(pop)]
        
        #append the list until it fills out nums
        count = 0
        while count < nums-1:
            word_idx = np.random.choice(len(pop), size=1)
            word = pop[word_idx[0]]
            if word.lower() in ENGLISH_FILTER_WORDS:
                continue

            bug = self.attack_cls.selectBug(word, if_initial=True)
            tokens = self.attack_cls.replaceWithBug(pop, word_idx[0], bug)
            #join it into a sentence
            x_prime_sent = " ".join(tokens)
            if (check_if_in_list(x_prime_sent, new_sent_ls)):
                continue

            new_sent_ls.append(x_prime_sent)
            new_pop.append(tokens)
            count += 1
            print("current count: ", count)
        
        return new_pop


    def get_fitness_score(self, input_tokens):
        #get fitness score of all the sentences
        sim_score_ls = []

        for tokens in input_tokens:
            x_prime_sent = " ".join(tokens)
            x_prime_sent = x_prime_sent.replace("_", " ")
            
            x_img_path = self.intermediate_path + "gen.png"
            
            dalle_mini_gen_img_from_text(x_prime_sent, x_img_path)

            similarity = compute_img_sim(x_img_path, self.target_img_path)

            #if similarity > self.store_thres:
            #    best_ori_path = self.best_img_path + x_prime_sent + "_score_" + str(similarity.item()) + ".png"
            #    shutil.copy(x_img_path, best_ori_path)

            sim_score_ls.append(similarity.item())

            print(f"x_prime_sent: {x_prime_sent}, similarity: {similarity.item()}")
        sim_score_arr = np.array(sim_score_ls)
        return sim_score_arr
    
    def mutate_pop(self, pop, mutation_p, mutate_by_impor):
        #random select the pop sentence that will mutate
        mask = np.random.rand(len(pop)) < mutation_p 
        new_pop = []
        pop_index = 0
        for flag in mask:
            if not flag:
                new_pop.append(pop[pop_index])
            else:
                tokens = pop[pop_index]
                
                if mutate_by_impor:
                    x_prime_sent = " ".join(tokens)
                    sim_probs = sort_words_by_importance(self.tokenizer, self.intermediate_path, x_prime_sent, self.target_img_path)
                    word_idx = np.random.choice(len(tokens), p=sim_probs, size=1)
                else:
                    word_idx = np.random.choice(len(tokens), size=1)
                word = tokens[word_idx[0]]

                if word.lower() in ENGLISH_FILTER_WORDS:
                    new_pop.append(pop[pop_index])
                    continue

                word_slice = word.split("_")
                if len(word_slice) > 1:
                    #randomly choose one
                    sub_word_idx = np.random.choice(len(word_slice), size=1)
                    sub_word = word_slice[sub_word_idx[0]]
                    bug = self.attack_cls.selectBug(sub_word, if_initial=False)
                    word_slice[sub_word_idx[0]] = bug
                    final_bug = '_'.join(word_slice)
                else:
                    final_bug = self.attack_cls.selectBug(word, if_initial=False)

                tokens = self.attack_cls.replaceWithBug(tokens, word_idx[0], final_bug)
                new_pop.append(tokens)
            pop_index += 1
        
        return new_pop
                    
    def run(self, log=None):
        best_save_dir = self.best_img_path
        itr = 1
        prev_score = None
        save_dir = self.intermediate_path
        best_score = float("-inf")
        if log is not None:
            log.write('target phrase: ' + self.target_sent + '\n')
        
        while itr <= self.max_iters:
            
            print(f"-----------itr num:{itr}----------------")
            log.write("------------- iteration:" + str(itr) + " ---------------\n")
            pop_scores = self.get_fitness_score(self.pop)
            elite_ind = np.argsort(pop_scores)[-self.elite_size:]
            elite_pop = [self.pop[i] for i in elite_ind]
            elite_pop_scores = pop_scores[elite_ind]

            print("current best score: ", elite_pop_scores[-1])
            
            for i in elite_ind:
                if pop_scores[i] > self.store_thres:
                    x_prime_sent_store = " ".join(self.pop[i])
                    x_prime_sent_store = x_prime_sent_store.replace("_", " ")
                    log.write(str(pop_scores[i]) + " " + x_prime_sent_store + "\n")
            
            if elite_pop_scores[-1] > best_score:
                best_score = elite_pop_scores[-1]
                #store the current best image
                x_prime_sent = " ".join(elite_pop[-1])
                x_prime_sent = x_prime_sent.replace("_", " ")
                
                x_img_path = save_dir + "gen.png"

                dalle_mini_gen_img_from_text(x_prime_sent, x_img_path)

                best_ori_path = best_save_dir + "itr_" + str(itr) + "_score_" + str(elite_pop_scores[-1]) + ".png"
                shutil.copy(x_img_path, best_ori_path)

                #new best adversarial sentences
                log.write("new best adv: " +  str(elite_pop_scores[-1]) + " " + x_prime_sent + "\n")
                log.flush()

            
            if prev_score is not None and prev_score != elite_pop_scores[-1]: 
                self.mutation_p = self.mu * self.mutation_p + self.alpha / np.abs(elite_pop_scores[-1] - prev_score) 

            next_pop = get_new_pop(elite_pop, elite_pop_scores, self.pop_size)

            self.pop = self.mutate_pop(next_pop, self.mutation_p, self.mutate_by_impor)
            
            prev_score = elite_pop_scores[-1]
            itr += 1

        return 
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_sent', type=str, required=True, help='original sentence')
    parser.add_argument('--tar_img_path', type=str, required=True, help='target image path')
    parser.add_argument('--tar_sent', type=str, required=True, help='target sentence')
    parser.add_argument('--log_save_path', type=str, default='run_log.txt', help='path to save log')
    parser.add_argument('--intem_img_path', type=str, default='./intermediate_img_path/', help='path to save intermediate imgs')
    parser.add_argument('--best_img_path', type=str, default='./best_img_path/', help='path to save best output imgs')
    parser.add_argument('--mutate_by_impor', type=bool, default=False, help='whether select word by importance in mutation')
    args = parser.parse_args()

    g = Genetic(args.ori_sent, args.tar_img_path, args.tar_sent, args.log_save_path, args.intem_img_path, args.best_img_path, args.mutate_by_impor)
    with open(args.log_save_path, 'w') as log:
        g.run(log=log)
