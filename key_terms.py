import string
import pandas as pd

import nltk
from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

with open('news_3.xml', 'r') as xml_news:
    tree = etree.parse(xml_news)
    root = tree.getroot()
    news = root[0]
    # etree.dump(news)

    headlines = []
    dataset = []

    for _news in news:
        news_dict = {}
        headlines.append(_news[0].text + ':')
        list_of_tokens = nltk.tokenize.word_tokenize(_news[1].text.lower())

    # start of stage 2

        list_of_stopwords = nltk.corpus.stopwords.words('english')
        list_of_punctuation = list(string.punctuation)

        list_of_lemmatized_tokens = []
        lemmatizer = nltk.WordNetLemmatizer()
        for token in list_of_tokens:
            list_of_lemmatized_tokens.append(lemmatizer.lemmatize(token))

        for token in list_of_lemmatized_tokens:
            if (token in list_of_stopwords) or (token in list_of_punctuation):
                pass     # end of stage 2

            # start of stage 3
            elif nltk.pos_tag([token])[0][1] != 'NN':
                pass    # end of stage 3
            else:
                if news_dict.get(token) is None:
                    news_dict[token] = 1
                else:
                    news_dict[token] = news_dict[token] + 1
        sorted_dict = sorted(news_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
        # print(sorted_dict)
        string_to_print = ''
        for _i in range(0, 5):
            string_to_print += str(sorted_dict[_i][0]) + ' '

        # start of stage 4
        string_to_vectorize = ''
        for key in sorted_dict:
            string_to_vectorize += str(key[0] + ' ') * key[1]
        dataset.append(string_to_vectorize)
        # print(string_to_print + '\\')

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset)
    terms = vectorizer.get_feature_names()
    df_terms = pd.DataFrame(terms, columns=['word']).reset_index()

    for i in range(len(headlines)):
        print(headlines[i])
        df = pd.DataFrame(tfidf_matrix[i].toarray())
        df2 = df.transpose().sort_values(by=0, ascending=False).reset_index()
        df3 = pd.merge(df2, df_terms, left_on='index', right_on='index').sort_values(by=[0, 'word'], ascending=[False, False])

        string_to_print = ''
        for _i in range(0, 5):
            part = terms[int(df3.iloc[_i]['index'])]
            string_to_print += part + ' '
        print(string_to_print + '\n')

    # end of stage 4


