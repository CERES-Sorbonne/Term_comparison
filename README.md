Main function : *update_dic_embeddings*
  """ 
  Input  : dic_embeddings, missing_terms
  Output : udpated  dic_embeddings according to missing_terms
  for each term of 'missing_terms':
    computes its embedding with sentence_transformer
    store it in dic_embeddings


