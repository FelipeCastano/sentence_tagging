import json

def load_tagged_sentences(filepath):
    """
    Load tagged sentences from a file where each sentence is separated by a blank line.
    Each line contains 'word tag' separated by space.
    Returns a list of sentences, each as list of (word, tag) tuples.
    """
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                word, tag = line.split()
                sentence.append((word, tag))
        # Add last sentence if file doesn't end with newline
        if sentence:
            sentences.append(sentence)
    return sentences

def load_word_index(filepath):
    """
    Load word-to-index dictionary from a JSON file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        word_to_index = json.load(f)
    return word_to_index


train_file = "../../data/train.txt"
corpus_file = "../../data/corpus.json"
train_sentences = load_tagged_sentences(train_file)
word_to_index = load_word_index(corpus_file)