from .requires import *
# import faiss
import numpy as np
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(np.array([t.cpu().numpy() for t in averaged_vectors]), np.array(range(0, documents_length)))


def encoder(document: str) -> torch.Tensor:
    tokens = transformer_tokenizer(document, return_tensors='pt').to(device)
    vector = transformer_model(**tokens)[0].detach().squeeze()
    return torch.mean(vector, dim=0)


def search(query: str, k=12):
    encoded_query = encoder(query).unsqueeze(dim=0).cpu().numpy()
    top_k = index.search(encoded_query, k)
    scores = top_k[0][0]
    return top_k[1][0]


def main(input):
    splitted_input = input.split(" ")
    nsi = []
    k = 12
    for x in splitted_input:
        if x not in stopwords:
            nsi.append(normalizer.normalize(lemmatizer.lemmatize(x)))
    input = " ".join(nsi)
    res = search(input, k)
    answer = []
    for x in res:
        temp = {'title': bioset[x]['title'], 'link': bioset[x]['link']}
        print(bioset[x]['title'])
        print(bioset[x]['link'])
        print()
        answer.append(temp)
    return answer


def retrieve(search_term):
    return main(search_term)
