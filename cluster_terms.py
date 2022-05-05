from sklearn.cluster import AffinityPropagation
import json
import tqdm
import warnings 
warnings.filterwarnings("ignore")

def update_dic_embeddings(dic_embeddings, missing_terms):
  """ 
  updates 'dic_embeddings' according to missing_terms
  for each term of 'missing_terms':
    computes its embedding with sentence_transformer
    store it in dic_embeddings
  """
  from sentence_transformers import SentenceTransformer, util
  EMBEDDING_MODEL = 'bert-base-multilingual-cased'
  model1 = SentenceTransformer(EMBEDDING_MODEL)
  for term in tqdm.tqdm(missing_terms):
    emb = model1.encode([term], convert_to_tensor=True)
    dic_embeddings[term] = [float(x) for x in list(emb)[0]]
  return dic_embeddings

def cluster_list_terms(list_terms):
  """ 
  Takes as entry a list of terms ('lis_terms')
  returns a clustering of the terms
  """
  path_dic_embeddings = "data/terms_embeddings.json"
  with open(path_dic_embeddings) as f:
    dic_embeddings = json.load(f)

  missing_terms = set(list_terms).difference(set(dic_embeddings.keys()))
  if len(missing_terms)>0:
    NB_missing = len(missing_terms)
    sample = sorted(list(missing_terms)[:10])
    print(f"Computing embeddings for {NB_missing} terms ({sample}...)")
    dic_embeddings = update_dic_embeddings(dic_embeddings, missing_terms)
    with open(path_dic_embeddings, "w") as w:
      w.write(json.dumps(dic_embeddings))
    print(f"Updated embeddings, stored in {path_dic_embeddings}")

  matrix = [dic_embeddings[x] for x in list_terms]
  clustering = AffinityPropagation(random_state=5).fit(matrix)
  dic_clusters = {}
  for i, lab in enumerate(list(clustering.labels_)):
    dic_clusters.setdefault(int(lab), [])
    dic_clusters[int(lab)].append(list_terms[i])
  return dic_clusters

if __name__=="__main__":
  print("Example")
  with open("data/terms.json") as f:
    liste = json.load(f)[:110]
  clusters =  cluster_list_terms(liste)
  print("-"*20)
  for cluster_name, cluster_terms in clusters.items():
    print(f"{cluster_name} :: {cluster_terms}")
