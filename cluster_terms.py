from sklearn.cluster import AffinityPropagation
import json
import tqdm
import warnings 
warnings.filterwarnings("ignore")

def update_dic_embeddings(dic_embeddings, missing_terms):
  from sentence_transformers import SentenceTransformer, util
  EMBEDDING_MODEL = 'bert-base-multilingual-cased'
  model1 = SentenceTransformer(EMBEDDING_MODEL)
  for term in tqdm.tqdm(missing_terms):
    emb = model1.encode([term], convert_to_tensor=True)
    dic_embeddings[term] = [float(x) for x in list(emb)[0]]
  return dic_embeddings

def cluster_list_terms(list_terms):
  with open("data/terms_embeddings.json") as f:
    dic_embeddings = json.load(f)
  missing_terms = set(list_terms).difference(set(dic_embeddings.keys()))
  if len(missing_terms)>0:
    NB_missing = len(missing_terms)
    sample = sorted(list(missing_terms)[:10])
    print(f"Computing embeddings for {NB_missing} terms ({sample}...)")
    dic_embeddings = update_dic_embeddings(dic_embeddings, missing_terms)
  matrix = [dic_embeddings[x] for x in list_terms]
  clustering = AffinityPropagation(random_state=5).fit(matrix)
  dic_clusters = {}
  for i, lab in enumerate(list(clustering.labels_)):
    dic_clusters.setdefault(int(lab), [])
    dic_clusters[int(lab)].append(list_terms[i])
  return dic_clusters

if __name__=="__main__":
  print("example")
  with open("data/terms.json") as f:
    liste = json.load(f)[:100]
  clusters =  cluster_list_terms(liste)
  for cluster_name, cluster_terms in clusters.items():
    print(f"{cluster_name} :: {cluster_terms}")
