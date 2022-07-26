import json
import numpy as np
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import nltk
nltk.download('punkt')

vectorizer = TfidfVectorizer(
    use_idf=True, norm='l2', ngram_range=(1, 2), stop_words='english')


def getLink(data):
    vectorizer.fit([' '.join(doc) for doc in data])
    X = vectorizer.transform([' '.join(doc) for doc in data])
    P = X.dot(X.T)
    P_norm = normalize(P, norm='l1')
    G = nx.from_numpy_matrix(P_norm.toarray())
    page_rank = nx.pagerank(G, alpha=0.9)
    return data[np.argmax(list(page_rank.values()))]


def getNutritionLink():
    data = None
    with open('all_nutrition_tokens_nonsymbol.json') as f:
        data = json.load(f)
    data = json.loads(data)
    return getLink(data)


def getDentalLink():
    data = None
    with open('all_dental_tokens_nonsymbol.json') as f:
        data = json.load(f)
    data = json.loads(data)
    return getLink(data)


def getFamilyHealthLink():
    data = None
    with open('all_family_health_tokens_nonsymbol.json') as f:
        data = json.load(f)
    data = json.loads(data)
    return getLink(data)


def getMentalHealthLink():
    data = None
    with open('all_mental_health_tokens_nonsymbol.json') as f:
        data = json.load(f)
    data = json.loads(data)
    return getLink(data)


def getNewInMedLink():
    data = None
    with open('all_new_in_med_tokens_nonsymbol.json') as f:
        data = json.load(f)
    data = json.loads(data)
    return getLink(data)


def getHairSkinLink():
    data = None
    with open('all_hair_skin_tokens_nonsymbol.json') as f:
        data = json.load(f)
    data = json.loads(data)
    return getLink(data)


def getProphylaxisDiseaseLink():
    data = None
    with open('all_prophylaxis_disease_tokens_nonsymbol.json') as f:
        data = json.load(f)
    data = json.loads(data)
    return getLink(data)


def getLinks():
    nutritionSentence = ''
    nutritionData = getNutritionLink()
    for word in nutritionData:
        nutritionSentence = nutritionSentence + ' ' + word
    nutritionSentence = nutritionSentence.strip()

    dentalSentence = ''
    dentalData = getDentalLink()
    for word in dentalData:
        dentalSentence = dentalSentence + ' ' + word
    dentalSentence = dentalSentence.strip()

    familyHealthSentence = ''
    familyHealthData = getFamilyHealthLink()
    for word in familyHealthData:
        familyHealthSentence = familyHealthSentence + ' ' + word
    familyHealthSentence = familyHealthSentence.strip()

    mentalHealthSentence = ''
    mentalHealthData = getMentalHealthLink()
    for word in mentalHealthData:
        mentalHealthSentence = mentalHealthSentence + ' ' + word
    mentalHealthSentence = mentalHealthSentence.strip()

    newInMedSentence = ''
    newInMedData = getNewInMedLink()
    for word in newInMedData:
        newInMedSentence = newInMedSentence + ' ' + word
    newInMedSentence = newInMedSentence.strip()

    hairSkinSentence = ''
    hairSkinData = getHairSkinLink()
    for word in hairSkinData:
        hairSkinSentence = hairSkinSentence + ' ' + word
    hairSkinSentence = hairSkinSentence.strip()

    prophylaxisDiseaseSentence = ''
    prophylaxisDiseaseData = getProphylaxisDiseaseLink()
    for word in prophylaxisDiseaseData:
        prophylaxisDiseaseSentence = prophylaxisDiseaseSentence + ' ' + word
    prophylaxisDiseaseSentence = prophylaxisDiseaseSentence.strip()

    return [nutritionSentence, dentalSentence, familyHealthSentence, mentalHealthSentence, newInMedSentence, hairSkinSentence, prophylaxisDiseaseSentence]


getLinks()
