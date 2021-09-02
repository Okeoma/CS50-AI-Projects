import nltk
import sys
import os
import string
import math

FILE_MATCHES = 4
SENTENCE_MATCHES = 3


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_content = {}

    for filename in os.listdir(directory):
        root = os.path.join(directory, filename)
        if filename.endswith(".txt"):
            if os.path.isfile(root):
                with open(root, "r", encoding='utf8') as f:
                    file_content[filename] = f.read()

    return file_content


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    content = nltk.word_tokenize(document.lower())
    content = [word for word in content if word not in nltk.corpus.stopwords.words("english")
               and word not in string.punctuation]

    return content


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    idf = {}
    num_docs = len(documents)
    for document in documents:
        words = set()
        for word_check in documents[document]:
            if word_check not in words:
                try:
                    idf[word_check] += 1
                except KeyError:
                    idf[word_check] = 1
                words.add(word_check)

    # calculate each word's idf
    for word in idf:
        idf[word] = math.log(num_docs / idf[word])
    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_names = {}

    for file in files:
        file_names[file] = 0
        for word in query:
            try:
                file_names[file] += files[file].count(word) * idfs[word]
            except KeyError:
                print(f"The word '{word}' in your query is not found in the documents provided")
                exit()

    file_names = sorted(file_names.keys(), key=lambda x: file_names[x], reverse=True)
    return file_names[0:n+1]


def sentence_idf(query_item, idfs):
    """
    Computes the idf value of words
    """
    idf_val = 0
    for word in query_item:
        idf_val += idfs[word]
    return idf_val


def sentence_qtd(query_item, words):
    """
    Computes the query term density of words
    """
    word_count = sum(map(lambda x: x in query_item, words))
    qtd_val = word_count / len(words)
    return qtd_val


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    scores = {}

    for sentence, words in sentences.items():
        # Collating the words that are common in the sentences
        query_item = query.intersection(words)

        # Evaluating idf value
        idf = sentence_idf(query_item,idfs)

        # Evaluating query term density value
        qtd = sentence_qtd(query_item, words)

        # update the list of sentences with idf and query term density values
        scores[sentence] = {'idf': idf, 'qtd': qtd}

    # Returns the ranked sentences by idf / query term density
    rank = sorted(scores.items(), key=lambda x: (x[1]['idf'], x[1]['qtd']), reverse=True)
    rank = [x[0] for x in rank]
    return rank[:n]

if __name__ == "__main__":
    main()
