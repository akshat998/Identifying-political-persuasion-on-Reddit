import sys
import argparse
import os
import json
import html
import re
import string

import timeit
import spacy

nlp = spacy.load('en', disable = ['parser','ner'])
#indir = '/u/cs401/A1/data/'
indir = '/Users/ouutsuyuki/cdf/csc401/data'

#abbrev_path = '/u/cs401/Wordlists/abbrev.english'
abbrev_path = '/Users/ouutsuyuki/cdf/csc401/Wordlists/abbrev.english'
abbrev = open(abbrev_path, 'r')
abbrev = abbrev.read().split('\n')

#StopWords_path = '/u/cs401/Wordlists/StopWords'
StopWords_path = '/Users/ouutsuyuki/cdf/csc401/Wordlists/StopWords'
StopWords = open(StopWords_path, 'r')
StopWords = StopWords.read().split('\n')


def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list 
        corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    step6 = 0       # Flag to check if part 8 is done
    step8 = 0       # Flag to check if part 8 is done

    #modComm = ''
    if 1 in steps:
        comment = comment.replace('\n','')

    if 2 in steps:
        comment = html.unescape(comment)        #replace &amp; to &feature, others worked fine

    if 3 in steps:
        comment = re.sub(r'(?:(?:http|https)?:\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&/=>]*)?',
            "", comment, flags=re.MULTILINE)
        # Source:  https://stackoverflow.com/questions/38804425/remove-urls-from-a-text-file

    if 4 in steps:

        word_list = re.compile("[\S]+").findall(comment)
        comment = ''
        for word in word_list:
            if not word in abbrev or word == 'e.g.' or word == 'i.e.':
                word = re.sub(r"[\W]+|[\w']+|[!\"#$%&\(\)*+,-./:;<=>?@[\]^_`{|}~\\]+", lambda pat: pat.group(0)+' ',
                                  word)
            comment += word

    if 5 in steps:
        # 's, 'd, 't, s', 've, 'll, 're, 'm
        comment = re.sub(r"([\w]+)(?=\'ve)|(?=\'ll)|(?=\'re)|(?=\'s)|(?=\'d)|(?=\'m)|(?=\'\s)",
                         lambda pat: pat.group(0) + ' ', comment, flags=re.I)
        comment = re.sub(r"([\w]+)(?=\'t)|(?=\'T)", lambda pat: pat.group(0)[:-1] + ' ' + pat.group(0)[-1], comment, flags=re.I)
        comment = comment.replace('\\','')

    if 6 in steps:
        #Tagging
        if step8 == 0:
            a = re.compile("[\S]+").findall(comment)
            doc = spacy.tokens.Doc(nlp.vocab,words = a)
            doc = nlp.tagger(doc)
            #utt = nlp(comment)
        comment = ''
        for token in doc:
            comment += str(token.text) + '/' + token.tag_ + ' '
        step6 = 0

    if 7 in steps:

        pattern = re.compile(r'\b(' + r'|'.join(StopWords) + r')\b')
        comment = pattern.sub('', comment)
        pattern = re.compile(r"\s/[\w]+(?=\s)")
        comment = pattern.sub('', comment)

    if 8 in steps:
        if step6 == 0:
            a = re.compile("([\w]+|[\W]+)/(?=[\w]+|[\W]+)").findall(comment)    # left word
            doc = spacy.tokens.Doc(nlp.vocab, words=a)
            doc = nlp.tagger(doc)
        for i in range(doc.__len__()):
            if str(doc[i]) in comment and doc[i].lemma_[0] != '-':
                comment = re.sub(re.escape(str(doc[i])), doc[i].lemma_, comment)
        step8 = 1

    if 9 in steps:
        comment = re.sub(r'/.(?=\s[A-Z])', lambda pat: pat.group(0) + '\n', comment)

    if 10 in steps:
        comment = re.sub(r'[\w]+(?=/)', lambda pat: pat.group(0).lower(), comment)

    modComm = comment
    return modComm

def main( args ):

    start = timeit.default_timer()

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            stop = timeit.default_timer()
            print(stop - start)

            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)
            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines

            lines = data[args.ID[0]%len(data) : args.ID[0]%len(data) + args.max]

            # TODO: read those lines with something like `j = json.loads(line)`

            for line in lines:          # line: string      lines: list
                j = json.loads(line)   # j: dict of a single line{}

            # TODO: choose to retain fields from those lines that are relevant to you

                desired_key = ['score', 'controversiality', 'author', 'body', 'id']
                #desired_key = ['author', 'body', 'id']
                j =({key:j[key] for key in desired_key})
                # dict_you_want = { your_key: old_dict[your_key] for your_key in your_keys }

            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)

                j['cat'] = file

            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument

                pre_text = preproc1(j['body'])

            # TODO: replace the 'body' field with the processed text

                j['body'] = pre_text

            # TODO: append the result to 'allOutput'
                allOutput.append(j)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

    # Count time
    stop = timeit.default_timer()
    print(stop - start)

    #print(allOutput)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    # 100 takes 40.334643396083266 seconds to run

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
        
    main(args)

