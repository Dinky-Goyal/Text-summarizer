# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:19:19 2020

@author: hp
"""

# Implementation from https://dev.to/davidisrawi/build-a-quick-summarizer-with-python-and-nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 



def calc_weighted_frequency(words,ps,lem,stopWords,text_string):
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    

    word_frequencies = dict()
    for word in words:
        word = ps.stem(word)
        word = lem.lemmatize(word)
        print(word)
        if word not in stopWords:
         if word not in word_frequencies:
            word_frequencies[word] = 1
         else:
            word_frequencies[word] += 1
            
    maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)        
    print(word_frequencies)
    return word_frequencies

def get_sentence_score(sentences, word_frequencies):
    """
    score a sentence by its words
    Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """
    sentence_scores = dict()
    for sent in sentences:
        word_count_without_stopwords=0
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
               word_count_without_stopwords+=1 
               if len(sent.split(' ')) < 30:
                   if sent not in sentence_scores.keys():
                      sentence_scores[sent] = word_frequencies[word]
                   else:
                      sentence_scores[sent] += word_frequencies[word]
    
        if sent in sentence_scores:
           sentence_scores[sent] = sentence_scores[sent]/word_count_without_stopwords
        
    print(sentence_scores)    
    return sentence_scores



def generate_summary(sentence_scores,lines):

  
    import heapq
    summary_sentences = heapq.nlargest(lines, sentence_scores, key=sentence_scores.get)
    print("\n")
    print(summary_sentences)
    print("\n")
    summary = ' '.join(summary_sentences)
    return summary

def run_summarized_text(text,lines):
    
    """ Removing stop words 
    Lemmatization - the process of grouping together the inflected forms of a word 
    so they can be analysed as a single item,
    Stemmer - an algorithm to bring words to its root word."""
 
    #text_preprocessing
    words = word_tokenize(text)
  #  print(words)
    print("\n")
    ps = PorterStemmer()
    lem = WordNetLemmatizer()
    stopWords = set(stopwords.words("english"))
   # print(stopWords)
    print("\n")
    # 1 Create the word frequency table
    freq_table = calc_weighted_frequency(words,ps,lem,stopWords,text)

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''

    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)
    print(sentences)
    print("\n")

    # 3 Important Algorithm: score the sentences
    sentence_scores = get_sentence_score(sentences, freq_table)

    #

    # 4 Important Algorithm: Generate the summary
    summary = generate_summary(sentence_scores,lines)

    return summary


if __name__ == '__main__':
    text_str = input("Enter the text: ") 
    print("\n")
    lines = int(input("Enter the number of lines you want in synopsis: "))
    print("\n")
    final_summary = run_summarized_text(text_str,lines)
    print("Synopsis of the text is given below:\n")
    print(final_summary)