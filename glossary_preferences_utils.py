import os
import pandas as pd
import numpy as np
import csv
import PyPDF2
from submodlib import SetCoverFunction

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

def get_txt(pdf_path) :
    pdfFileObj = open(pdf_path, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    txt = "" 
    for page in range(pdfReader.numPages) : 
        pageObj = pdfReader.getPage(page)
        txt += pageObj.extractText()

    pdfFileObj.close()
    return txt

def get_dict_words(path, src_lang, trans_lang) :
  dict_words = dict()
  files = os.listdir(path)
#   files = [file_ for file_ in files if file_.endswith(src_lang + "_" + trans_lang + ".csv")]
  for file_ in files :
    try :
      df = pd.read_csv(path + "/" + file_)
      words = df.loc[:,"English"]
    except : 
      continue
    dict_words[file_] = []
    for word in words :
      try :
        dict_words[file_].append(word.lower())
      except :
        continue
  
  return dict_words

def get_tokens(txt, tokenizer, lemmatizer, stop_words) :
    tokenized = tokenizer.tokenize(txt)
    statement_no_stop = [word.lower() for word in tokenized if word.lower() not in stop_words]
    lemmatized = [lemmatizer.lemmatize(token) for token in statement_no_stop]

    return lemmatized

def get_idfs_samanantar(tokens) :
    csv_reader = csv.reader(open('./idf_samananthar.csv', 'r'))
    idf = dict()
    for row in csv_reader:
        k, v = row
        idf[k] = float(v)

    unknown_idf = max(list(idf.values()))
    weights = [idf[token] if token in idf else unknown_idf for token in tokens ]
    
    return weights

def get_coverage(dict_words, tokens, tokenizer, lemmatizer, stop_words) :
    tokens = np.array(tokens)
    coverage = []

    for term in dict_words :
        tokenized = [lemmatizer.lemmatize(word.lower()) for word in tokenizer.tokenize(term) if word.lower() not in stop_words]
        if len(tokenized) == 0 :
            continue
        try :
            matched_inds = np.where(tokens == tokenized[0])[0]
        except :
            continue
        
        if len(tokenized) == 1 :
            coverage += list(matched_inds)
        
        else :
            for ind in matched_inds :
                discard = False
                for j in range(1, len(tokenized)) :
                    if tokens[min(ind + j, len(tokens) - 1)] != tokenized[j] :
                        discard = True
                        break
                if not discard :
                    coverage += list(range(ind, ind + len(tokenized)))  

    return set(coverage)

def get_set_cover(source_text, all_dict_words, budget) :
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer()

    tokens = get_tokens(source_text, tokenizer, lemmatizer, stop_words)
    weights = get_idfs_samanantar(tokens)
    
    dict_coverage = []
    for name, dict_words in all_dict_words.items() :
        dict_coverage.append(get_coverage(dict_words, tokens, tokenizer, lemmatizer, stop_words))

    n=len(all_dict_words)
    if budget >= n :
        budget = n - 1

    obj = SetCoverFunction(n=n, cover_set=dict_coverage, num_concepts=len(tokens), concept_weights = weights)
    greedyList = obj.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

    return greedyList

# src_lang and trans_lang can be used to filter the relevant glossaries from which we should select
# currently working only with en to hi, ignoring these arguments
def select_glossaries(pdf_path, src_lang, trans_lang, glossaries_path) :
    if (src_lang != "en" or trans_lang != "hi") :
        return None
    
    txt = get_txt(pdf_path)
    all_dict_words = get_dict_words(glossaries_path, src_lang, trans_lang)

    set_cover = get_set_cover(txt, all_dict_words, len(all_dict_words))
    dictionaries = list(all_dict_words.keys())
    selected_dictionaries = [dictionaries[_[0]][:-4] for _ in set_cover]
    
    return ",".join(selected_dictionaries)