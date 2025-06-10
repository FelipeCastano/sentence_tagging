import numpy as np 
import string 
# Punctuation characters
punct = set(string.punctuation)

# Morphology rules used to assign unknown word tokens
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]

espanish_noun_suffix = ["ción", "sión", "dad", "tud", "ez", "eza", "aje", "al", "or", "ista", "ero", "ía", "ismo", "ezno", "umbre"]
espanish_verb_suffix = ["ar", "er", "ir", "ecer", "izar", "ificar"]
espanish_adj_suffix = ["able", "ible", "oso", "al", "ico", "il", "ar", "ente", "ivo", "ario", "az", "ón"]
espanish_adv_suffix = ["mente"]

def assign_unk_espanish(tok):
    """
    Assign unknown word tokens
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in espanish_noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in espanish_verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in espanish_adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in espanish_adv_suffix):
        return "--unk_adv--"
    return "--unk--"

def assign_unk_english(tok):
    """
    Assign unknown word tokens
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"

def load_data(path, is_corpus):
    '''
    Loads data from a given file.

    Input:
        path: path to the file
        is_corpus: boolean flag to determine if the file is a corpus (for vocab creation)
    Output:
        If is_corpus is True: returns a vocabulary dictionary mapping word -> index
        If is_corpus is False: returns the file lines as a list
    '''
    with open(path, 'r') as f:
        if is_corpus:
            vocab = {}
            file = f.read().split('\n')
            for i, word in enumerate(sorted(file)):
                vocab[word] = i
            return vocab
        else:
            file = [line for line in f.readlines()]
            return file

def get_word_tag(line, vocab, english): 
    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocab: 
            # Handle unknown words
            if(english):
                word = assign_unk_english(word)
            else:
                word = assign_unk_espanish(word)
        return word, tag
    return None 
