from .requires import *

num_of_words = []


def splitInput(input):
    splitted_input = input.split(" ")
    nsi = []
    for x in splitted_input:
        if x not in stop_words and len(x) > 2:
            nsi.append(normalizer.normalize(lemmatizer.lemmatize(x)))
    return nsi


def boolean_retrieval():
    """
    :param corpus: a list of documents
    :return: a list of documents that match the query
    """
    global num_of_words

    num_of_words = []

    bag_of_words = []
    unique_words = set()
    lemmatized_docs = [' '.join(x) for x in all_tokens_nonstop_lemstem]
    for doc in lemmatized_docs:
        temp = doc.split(' ')
        bag_of_word = [t for t in temp if len(t) > 2]
        bag_of_words.append(bag_of_word)
        unique_words = set(unique_words).union(set(bag_of_word))

    for i in range(len(bag_of_words)):
        numOfWord = dict.fromkeys(unique_words, 0)
        for word in bag_of_words[i]:
            numOfWord[word] += 1
        num_of_words.append(numOfWord)


def boolean_query(query, k=10):
    score = {}
    for i in range(len(all_tokens_nonstop_lemstem)):
        score[i] = 0

    nsi = splitInput(query)
    counter = 0
    for doc in num_of_words:
        for word in nsi:
            try:
                if doc[word] > 0:
                    score[counter] += 1
                    # you can comment the next line
                    score[counter] += 0.02 * doc[word]
            except:
                pass
        counter += 1
    sorted_score = {k: v for k, v in sorted(
        score.items(), key=lambda item: item[1], reverse=True)}

    counter = 0
    res = []
    for x in sorted_score:
        if counter >= k:
            break
        # temp = {'title': bioset[x]['title'], 'link': bioset[x]['link']}
        temp = '' + bioset[x]['title'] + '\n' + bioset[x]['link']

        # print(bioset[x]['title'])
        # print(bioset[x]['link'])
        # print()

        counter += 1
        res.append(temp)

    return res


print("training boolean_retrieval retrieval system for health")
boolean_retrieval()
print("training boolean_retrieval retrieval system for health done")


def retrieve(search_term):
    return boolean_query(search_term)
