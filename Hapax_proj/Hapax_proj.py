from collections import Counter
import re

BOM_path = "C:\\Users\\jbren\\Desktop\\Textbooks\\LING_240_Readings\\Hapax_proj\\The_Book_of_Mormon.txt"
with open(BOM_path, 'r', encoding='utf-8') as file:
    BOM_text = file.read()

# Function to tokenize text into words while eliminating non-letter characters
def tokenize(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words

# Function to find hapax legomena in a corpus
def find_hapax_legomena(corpus_text):
    corpus_words = tokenize(corpus_text)
    word_freq = Counter(corpus_words)
    hapax_legomena = [word for word, freq in word_freq.items() if freq == 1]
    return hapax_legomena

# List of corpus files (replace with your file paths)
corpus_files = ['BOM_txt']#, 'corpus2.txt', 'corpus3.txt']

# Iterate through each corpus
for i, corpus_file in enumerate(corpus_files, start=1):
    with open(corpus_file, 'r', encoding='utf-8') as file:
        corpus_text = file.read()
    
    # Find hapax legomena in the current corpus
    hapax_legomena_in_corpus = find_hapax_legomena(corpus_text)
    
    # Print hapax legomena in the current corpus
    print(f"Hapax Legomena in Corpus {i} ({corpus_file}):")
    for word in hapax_legomena_in_corpus:
        print(word)
    
    # Update the global list of hapax legomena across all corpora
    if i == 1:
        hapax_legomena_across_all_corpora = set(hapax_legomena_in_corpus)
    else:
        hapax_legomena_across_all_corpora &= set(hapax_legomena_in_corpus)

# Print hapax legomena across all three corpora
print("\nHapax Legomena Across All Three Corpora:")
for word in hapax_legomena_across_all_corpora:
    print(word)

