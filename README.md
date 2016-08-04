# Promoter-Discovery

To extract training regions from the genome, use the following code:
`python extract_regions.py > regions.txt`
The output file, regions.txt, contains lines of the genome sequence and label (promoter or enhancer).

To run the classification analysis, use the following code:
`python analysis.py`
(You'll need to install the gensim library, see [https://radimrehurek.com/gensim/install.html](https://radimrehurek.com/gensim/install.html).)

