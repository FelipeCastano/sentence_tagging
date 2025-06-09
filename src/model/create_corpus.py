import os
import json
from itertools import chain
from sklearn.model_selection import train_test_split

def save_sentences(sent_list, filepath):
    """
    Save a list of tagged sentences (or tokens) to a file.
    Each word and tag is written in a separate line separated by space,
    and an empty line separates sentences (if input is list of lists).
    For flat lists (list of tokens), it writes all tokens consecutively without blank lines.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        # Check if input is a list of sentences (list of lists) or flat list
        if sent_list and isinstance(sent_list[0], list):
            # list of sentences
            for sent in sent_list:
                for word, tag in sent:
                    f.write(f"{word} {tag}\n")
                f.write("\n")
        else:
            # flat list of tokens
            for word, tag in sent_list:
                f.write(f"{word} {tag}\n")

def load_tagged_sentences(filepath):
    """
    Load tagged sentences from a file where each line is a sentence
    with tokens in 'word/tag' format separated by spaces.
    Returns a list of sentences, each as a list of (word, tag) tuples.
    """
    sentences = []
    words = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word_tag_pairs = line.split()
            sentence = []
            for pair in word_tag_pairs:
                if '/' in pair:
                    idx = pair.rfind('/')
                    word = pair[:idx]
                    tag = pair[idx+1:]
                    token = (word, tag)
                    sentence.append(token)  # Append token to the sentence
                    words.append(word)
            sentences.append(sentence)
    return sentences, words

def build_word_index(sentences):
    """
    Extract unique words from the training set,
    add special tokens, and create a dictionary {word: index}.
    """
    train_words = []
    for sentence in sentences:
        for word, _ in sentence:
            train_words.append(word)
    train_words = list(set(train_words))

    unk_words = [
        "--unk_digit--", "--unk_punct--", "--unk_upper--",
        "--unk_noun--", "--unk_verb--", "--unk_adj--", "--unk_adv--", "--unk--"
    ]
    train_words.extend(unk_words)
    train_words = sorted(train_words)
    word_to_index = {word: idx for idx, word in enumerate(train_words)}
    return word_to_index

def save_word_index(word_to_index, filepath):
    """
    Save the word_to_index dictionary to a JSON file with readable formatting.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(word_to_index, f, ensure_ascii=False, indent=4)

# --- Main execution ---

INPUT_FILE = "data/wsj_tagged.txt"
OUTPUT_DIR = "data"

# Load tagged sentences
sentences, _ = load_tagged_sentences(INPUT_FILE)

# Split into training and test sets (80% / 20%)
train_sents, test_sents = train_test_split(sentences, test_size=0.2, random_state=42)

# Flatten lists of lists into single list of tokens for each set
train_sents = list(chain.from_iterable(train_sents))
test_sents = list(chain.from_iterable(test_sents))

# Save the splits to files (this will save as flat lists, no blank lines between sentences)
save_sentences(train_sents, os.path.join(OUTPUT_DIR, "train.txt"))
save_sentences(test_sents, os.path.join(OUTPUT_DIR, "test.txt"))

# Build the word-to-index dictionary including special tokens
# Note: build_word_index expects list of sentences, so wrap tokens back into sentences of length 1 for compatibility or adapt the function
# Here we adapt by grouping tokens into sentences of length 1:
train_sentences_wrapped = [[token] for token in train_sents]
word_to_index = build_word_index(train_sentences_wrapped)

# Save the dictionary as JSON
save_word_index(word_to_index, os.path.join(OUTPUT_DIR, 'corpus.json'))
