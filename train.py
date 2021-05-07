!pip install sklearn-crfsuite
import sklearn_crfsuite

from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
import numpy as np



def main():
  dim = 0
  word_vecs= {}
  # open pretrained word vector file
  fileName = '/content/drive/MyDrive/AIMAS/Project3/cna.cbow.cwe_p.tar_g.512d.0.txt'
  #fileName = '/content/drive/MyDrive/AIMAS/Project3/zh_wiki_fasttext_300.txt'
  with open(fileName) as f:
    for line in f:
      tokens = line.strip().split()

      # there 2 integers in the first line: vocabulary_size, word_vector_dim
      if len(tokens) == 2:
        dim = int(tokens[1])
        continue

      word = tokens[0]
      vec = np.array([ float(t) for t in tokens[1:] ])
      word_vecs[word] = vec
  print('vocabulary_size: ',len(word_vecs),' word_vector_dim: ',vec.shape)

  data_path='sample.data'
  data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = Dataset(data_path)
  trainembed_list = Word2Vector(traindata_list, word_vecs)
  testembed_list = Word2Vector(testdata_list, word_vecs)

  # CRF - Train Data (Augmentation Data)
  x_train = Feature(trainembed_list)
  y_train = Preprocess(traindata_list)

  # CRF - Test Data (Golden Standard)
  x_test = Feature(testembed_list)
  y_test = Preprocess(testdata_list)

  #new
  #x_train = [sent2features(s) for s in trainembed_list]
  #y_train = Preprocess(traindata_list)

  #x_test = [sent2features(s) for s in testembed_list]
  #y_test = Preprocess(testdata_list)

  y_pred, y_pred_mar, f1score = CRF(x_train, y_train, x_test, y_test)

  print(f1score)
  #print(len(y_pred[0]))
  output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
  for test_id in range(len(y_pred)):
    pos=0
    start_pos=None
    end_pos=None
    entity_text=None
    entity_type=None
    for pred_id in range(len(y_pred[test_id])):
      if y_pred[test_id][pred_id][0]=='B':
        start_pos=pos
        entity_type=y_pred[test_id][pred_id][2:]
      elif start_pos is not None and y_pred[test_id][pred_id][0]=='I' and y_pred[test_id][pred_id+1][0]=='O':
        end_pos=pos
        entity_text=''.join([testdata_list[test_id][position][0] for position in range(start_pos,end_pos+1)])
        line=str(testdata_article_id_list[test_id])+'\t'+str(start_pos)+'\t'+str(end_pos+1)+'\t'+entity_text+'\t'+entity_type
        output+=line+'\n'
      pos+=1

  output_path='output.tsv'
  with open(output_path,'w',encoding='utf-8') as f:
    f.write(output)

  print(output)

def CRF(x_train, y_train, x_test, y_test):
  crf = sklearn_crfsuite.CRF(
      algorithm='lbfgs',
      c1=0.1,
      c2=0.1,
      max_iterations=100,
      all_possible_transitions=True
  )
  crf.fit(x_train, y_train)
  # print(crf)
  y_pred = crf.predict(x_test)
  y_pred_mar = crf.predict_marginals(x_test)

  # print(y_pred_mar)

  labels = list(crf.classes_)
  labels.remove('O')
  f1score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
  sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0])) # group B and I results

  print(flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))

  #write prediction output to development.data
  output_path1 = 'development.data'
  with open(output_path1,'w',encoding='utf-8') as f:
    f.write(flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
  return y_pred, y_pred_mar, f1score


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def word2features(sent, i):
    word = sent[i][0]
    word = str(word)
    postag = sent[i][1]
    postag = str(postag)
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        word1 = str(word1)
        postag1 = sent[i-1][1]
        postag1= str(postag1)
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        word1 = str(word1)
        postag1= str(postag1)
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features

def Dataset(data_path):
  with open(data_path, 'r', encoding='utf-8') as f:
    data=f.readlines()#.encode('utf-8').decode('utf-8-sig')
  data_list, data_list_tmp = list(), list()
  article_id_list=list()
  idx=0
  for row in data:
    data_tuple = tuple()
    if row == '\n':
      article_id_list.append(idx)
      idx+=1
      data_list.append(data_list_tmp)
      data_list_tmp = []
    else:
      row = row.strip('\n').split(' ')
      data_tuple = (row[0], row[1])
      data_list_tmp.append(data_tuple)
  if len(data_list_tmp) != 0:
    data_list.append(data_list_tmp)
  
  # here we random split data into training dataset and testing dataset
  # but you should take `development data` or `test data` as testing data
  # At that time, you could just delete this line,
  # and generate data_list of `train data` and data_list of `development/test data` by this function
  traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list=train_test_split(data_list,
                                                                                                  article_id_list,
                                                                                                  test_size=0.33,
                                                                                                  random_state=42)
  
  return data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list

def Word2Vector(data_list, embedding_dict):
  embedding_list = list()

  # No Match Word (unknown word) Vector in Embedding
  unk_vector=np.random.rand(*(list(embedding_dict.values())[0].shape))

  for idx_list in range(len(data_list)):
    embedding_list_tmp = list()
    for idx_tuple in range(len(data_list[idx_list])):
      key = data_list[idx_list][idx_tuple][0] # token

      if key in embedding_dict:
          value = embedding_dict[key]
      else:
          value = unk_vector
      embedding_list_tmp.append(value)
    embedding_list.append(embedding_list_tmp)
  return embedding_list

def Feature(embed_list):
  feature_list = list()
  for idx_list in range(len(embed_list)):
    feature_list_tmp = list()
    for idx_tuple in range(len(embed_list[idx_list])):
      feature_dict = dict()
      for idx_vec in range(len(embed_list[idx_list][idx_tuple])):
        feature_dict['dim_' + str(idx_vec+1)] = embed_list[idx_list][idx_tuple][idx_vec]
      feature_list_tmp.append(feature_dict)
    feature_list.append(feature_list_tmp)
  return feature_list

def Preprocess(data_list):
  label_list = list()
  for idx_list in range(len(data_list)):
    label_list_tmp = list()
    for idx_tuple in range(len(data_list[idx_list])):
      label_list_tmp.append(data_list[idx_list][idx_tuple][1])
    label_list.append(label_list_tmp)
  return label_list

if __name__ == '__main__':
  main()
