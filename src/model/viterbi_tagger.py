import json
import string
import numpy as np
import math
from collections import defaultdict
from utils import get_word_tag 

class ViterbiTagger:
    '''
    Encapsulates the Viterbi algorithm and helper utilities for POS tagging.
    Existing docstrings have been preserved verbatim. Any new comments are in English.
    '''

    def __init__(self, alpha: float = 0.001):
        '''
        Initializes the tagger with a smoothing factor.

        Input:
            alpha: smoothing factor for add‑alpha smoothing in probability matrices
        '''
        self.alpha = alpha

    def create_dictionaries(self, training_corpus, vocab, verbose=True):
        """
        Input: 
            training_corpus: a corpus where each line has a word followed by its tag.
            vocab: a dictionary where keys are words in vocabulary and value is an index
        Output: 
            emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
            transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
            tag_counts: a dictionary where the keys are the tags and the values are the counts
        """
        emission_counts = defaultdict(int)
        transition_counts = defaultdict(int)
        tag_counts = defaultdict(int)
        prev_tag = '--s--' 
        i = 0 
        for word_tag in training_corpus:
            i += 1
            if i % 50000 == 0 and verbose:
                print(f"word count = {i}")
            word, tag = get_word_tag(word_tag, vocab)
            transition_counts[(prev_tag, tag)] += 1
            emission_counts[(tag, word)] += 1
            tag_counts[tag] += 1
            prev_tag = tag
    
        self.emission_counts = emission_counts
        self.transition_counts = transition_counts
        self.tag_counts = tag_counts
    
    @staticmethod
    def create_transition_matrix(alpha, tag_counts, transition_counts):
        ''' 
        Creates the transition probability matrix A using add-alpha smoothing.

        Input: 
            alpha: smoothing factor
            tag_counts: dict mapping each tag to its count
            transition_counts: dict with (prev_tag, tag) tuples as keys and their counts as values
        Output:
            A: transition probability matrix (num_tags x num_tags)
        '''
        all_tags = sorted(tag_counts.keys())
        num_tags = len(all_tags)
        A = np.zeros((num_tags, num_tags))
        trans_keys = set(transition_counts.keys())

        for i in range(num_tags):
            for j in range(num_tags):
                count = 0
                key = (all_tags[i], all_tags[j])
                if key in trans_keys:
                    count = transition_counts[key]
                count_prev_tag = tag_counts[key[0]]
                A[i, j] = (count + alpha) / (count_prev_tag + (alpha * num_tags))

        return A

    @staticmethod
    def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
        '''
        Creates the emission probability matrix B using add-alpha smoothing.

        Input: 
            alpha: smoothing factor
            tag_counts: dict mapping each tag to its count
            emission_counts: dict with (tag, word) tuples as keys and their counts as values
            vocab: dict mapping word -> index
        Output:
            B: emission probability matrix (num_tags x vocab_size)
        '''
        num_tags = len(tag_counts)
        all_tags = sorted(tag_counts.keys())
        num_words = len(vocab)
        B = np.zeros((num_tags, num_words))
        emis_keys = set(emission_counts.keys())

        for i in range(num_tags):
            for j in range(num_words):
                count = 0 
                key = (all_tags[i], vocab[j])
                if key in emis_keys:
                    count = emission_counts[key]
                count_tag = tag_counts[key[0]]
                B[i, j] = (count + alpha) / (count_tag + (num_words * alpha))

        return B

    @staticmethod
    def initialize(states, tag_counts, A, B, corpus, vocab):
        '''
        Initializes the Viterbi matrices for dynamic programming.

        Input: 
            states: list of all POS tags
            tag_counts: dict mapping tag -> count
            A: transition matrix
            B: emission matrix
            corpus: list of words
            vocab: dict mapping word -> index
        Output:
            best_probs: matrix of log probabilities
            best_paths: matrix of backpointers
        '''
        num_tags = len(tag_counts)
        best_probs = np.zeros((num_tags, len(corpus)))
        best_paths = np.zeros((num_tags, len(corpus)), dtype=int)
        s_idx = states.index("--s--")

        for i in range(num_tags):
            best_probs[i, 0] = math.log(A[s_idx, i]) + math.log(B[i, vocab[corpus[0]]])

        return best_probs, best_paths

    @staticmethod
    def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab, verbose=True):
        '''
        Performs the forward step of the Viterbi algorithm.

        Input:
            A, B: transition and emission matrices
            test_corpus: list of words
            best_probs: initialized log-probability matrix
            best_paths: initialized backpointer matrix
            vocab: dict mapping word -> index
            verbose: flag to print progress
        Output:
            best_probs: completed log-probability matrix
            best_paths: completed backpointer matrix
        '''
        num_tags = best_probs.shape[0]

        for i in range(1, len(test_corpus)):
            if i % 5000 == 0 and verbose:
                print("Words processed: {:>8}".format(i))

            for j in range(num_tags):
                best_prob_i = float("-inf")
                best_path_i = None

                for k in range(num_tags):
                    prob = best_probs[k, i - 1] + math.log(A[k, j]) + math.log(B[j, vocab[test_corpus[i]]])
                    if prob > best_prob_i:
                        best_prob_i = prob
                        best_path_i = k

                best_probs[j, i] = best_prob_i
                best_paths[j, i] = best_path_i

        return best_probs, best_paths

    @staticmethod
    def viterbi_backward(best_probs, best_paths, corpus, states):
        '''
        Performs the backward step of the Viterbi algorithm.

        Input:
            best_probs: log-probability matrix
            best_paths: backpointer matrix
            corpus: list of words
            states: list of all POS tags
        Output:
            pred: list of predicted POS tags
        '''
        m = best_paths.shape[1]
        z = [None] * m
        num_tags = best_probs.shape[0]
        best_prob_for_last_word = float('-inf')
        pred = [None] * m

        for k in range(num_tags):
            if best_probs[k, -1] > best_prob_for_last_word:
                best_prob_for_last_word = best_probs[k, -1]
                z[m - 1] = k

        pred[m - 1] = states[z[m - 1]]

        for i in range(m - 1, 0, -1):
            pos_tag_for_word_i = z[i]
            z[i - 1] = best_paths[pos_tag_for_word_i, i]
            pred[i - 1] = states[z[i - 1]]

        return pred

    @staticmethod
    def predict_pos(prep, y, emission_counts, vocab, states):
        '''
        Performs a naive prediction using the emission counts.

        Input:
            prep: list of words from the corpus
            y: original labeled corpus as (word POS) strings
            emission_counts: dict with (tag, word) tuples as keys
            vocab: word to index mapping
            states: list of all POS tags
        Output:
            accuracy: float indicating prediction accuracy
        '''
        num_correct = 0
        all_words = set(emission_counts.keys())
        total = 0

        for word, y_tup in zip(prep, y): 
            y_tup_l = y_tup.split()
            if len(y_tup_l) == 2:
                true_label = y_tup_l[1]
            else:
                continue
        
            count_final = 0
            pos_final = ''

            if word in vocab:
                for pos in states:
                    key = (pos, word)
                    if key in all_words:
                        count = emission_counts[key]
                        if count > count_final:
                            count_final = count
                            pos_final = pos

                if pos_final == true_label:
                    num_correct += 1

            total += 1 
            
        accuracy = num_correct / total
        return accuracy

    @staticmethod
    def compute_accuracy(pred, y):
        '''
        Computes the accuracy of predictions against ground truth.

        Input:
            pred: list of predicted POS tags
            y: list of labeled lines ("word tag")
        Output:
            accuracy: float
        '''
        num_correct = 0
        total = 0

        for prediction, y in zip(pred, y):
            word_tag_tuple = y.split()
            if len(word_tag_tuple) != 2:
                continue
            word, tag = word_tag_tuple[0], word_tag_tuple[1]
            if prediction == tag:
                num_correct += 1
            total += 1

        return num_correct / total

    def build_matrices(self, training_corpus, vocab):
        '''
        Builds and stores the transition (A) and emission (B) matrices using the tagger's alpha value.
    
        Input:
            tag_counts: dict mapping each tag to its count
            transition_counts: dict with (prev_tag, tag) tuples as keys and their counts as values
            emission_counts: dict with (tag, word) tuples as keys and their counts as values
            vocab: dict mapping word -> index
    
        Output:
            None (matrices are stored internally in self.A and self.B)
        '''
        self.create_dictionaries(training_corpus, vocab)
        self.A = self.create_transition_matrix(self.alpha, self.tag_counts, self.transition_counts)
        self.B = self.create_emission_matrix(self.alpha, self.tag_counts, self.emission_counts, vocab)
        self.vocab = vocab
        self.states = sorted(tag_counts.keys())


    def tag(self, corpus):
        '''
        Performs POS tagging on the given corpus using the internally stored matrices.
        
        Input:
            corpus: list of words (preprocessed)
        
        Output:
            pred: list of predicted POS tags corresponding to the input words
        '''
        if self.A is None or self.B is None:
            raise ValueError("Matrices A and B have not been built. Call build_matrices() first.")
        best_probs, best_paths = self.initialize(
            self.states,
            defaultdict(int, {s: i for i, s in enumerate(self.states)}),
            self.A, self.B, corpus, self.vocab
        )
        best_probs, best_paths = self.viterbi_forward(self.A, self.B, corpus, best_probs, best_paths, self.vocab, verbose=False)
        pred = self.viterbi_backward(best_probs, best_paths, corpus, self.states)
        return pred
