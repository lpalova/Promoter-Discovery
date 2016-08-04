# Promoter-Discovery

To extract training regions from the genome, use the following code:
`python extract_regions.py > regions.txt`
The output file, regions.txt, contains lines of the (1000 character) genome sequences with labels (promoter or enhancer).

To run the classification analysis, use the following code:
`python analysis.py`
(You'll need to install the gensim library, see [https://radimrehurek.com/gensim/install.html](https://radimrehurek.com/gensim/install.html).)

Parameters and details in the classification analysis:

(i) Parameters *min_word_length* and *max_word_length* correspond to the minimum and maximum length of a word in the sequence. Word is any sequence of A/C/G/T characters. Here we use *min_word_length* of 1 and *max_word_length* of 4, and we assume that we can interpret any sequence longer than 4 by a combination 1-, 2-, 3-, or 4-characters.
The other limits are given by the maximum length of a sequence (*max_text_length* of 1000 characters) and the number of sequences, sampled out of the post-processed *train_data* files, parameter in the *sample_input* function (e.g., 10000).

(ii) The function *process_text* processes a list of sequences. For each sequence, it creates a dictionary with keys corresponding to words (a word is any part of the sequence of a given length) and values corresponding to the word frequency (counter). Index (integer) to key (word) mapping is done via the *ind_to_key* and *key_to_ind* dictionaries. 
The *get_corpus* function changes the list of dictionaries to a list of tuples format of the processed sequences. 
The *get_Xy* function's input is corpus of sequences (text) together with a vector of sequences' labels, and a dimensionality reduction parameter (ndims). The text is processed using tfidf weighting and lsi (as SVD dimensionality reduction with ndims dimensions). The output is document-"topic" (term) matrix X and vector of labels (0 or 1) y.

(iii) Classification is implemented using logistic regression. An average F1-score is calculated using 10 fold cross validation. For insight, the classification report, confusion matrix and the F1-score are computed for one train-test split. The average F1 score is about 0.66.
