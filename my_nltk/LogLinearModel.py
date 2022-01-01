from typing import Generator, Iterable, List

import numpy as np

class LLM():
  """対数線形分類モデル"""
  def __init__(self, eta, c, stop_num) -> None:
    self.eta = eta
    self.c = c
    self.stop_num = stop_num
    self.L = -999

  def fit(self, docs:List, labels: List):
    """学習を行う

    Args:
      docs (List): 入力文
      abels (List): ラベル
    """

    self.docs = np.array(docs)
    self.labels = np.array(labels)
    self.classes = set(labels)

    # 特徴量を作成(語彙とラベルの直積集合)
    self.fearutures = [f for f in self._build_feature()]

    # 文章を特徴量ベクトルに変換
    self.fm = np.array([fm for fm in self._doc2fvec(self.docs)])
    self.w = np.zeros(self.fm.shape[2])
    dif_ = 1
    while self.stop_num < dif_:
      self.prob = self._calc_prob(self.fm)
      w_ = self._updateW()
      self.w = self.w + self.eta * w_
      current_L = self._negative_log_likelihood()
      dif_ = current_L - self.L
      self.L = current_L

  def predict(self, docs):
    """推測"""
    fm = np.array([fm for fm in self._doc2fvec(docs)])
    prob = self._calc_prob(fm)
    return np.argmax(prob, axis=0)

  def _build_feature(self)->Generator:
    """特徴量を作成"""
    vocab = set([word for sent in self.docs for word in sent.strip().split()])
    added = set()

    for word in vocab:
      for label in self.labels:
        if word + str(label) in added:
          pass
        else:
          added.add(word + str(label))
          yield (word, label)

  def _doc2fvec(self, docs:Iterable[str]):
    """文章を特徴量ベクトルに変換"""
    for i, doc in enumerate(docs):
      label_list = []
      for label in self.classes:
        feature_list = []
        for feature in self.fearutures:
          if feature[0] in doc.split() and feature[1] == label:
            feature_list.append(1)
          else:
            feature_list.append(0)
        label_list.append(feature_list)
      yield label_list

  def _calc_prob(self, fm):
    dot_ = np.dot(fm, self.w)
    return self._softmax(dot_)

  def _softmax(self, f):
    e = np.exp(f)
    return e / np.sum(e, axis=1, keepdims=True)

  def _updateW(self):
    sum_f = np.zeros(self.fm.shape[2])
    for i, (sent, label) in enumerate(zip(self.docs, self.labels)):
      sum_f = sum_f + self.fm[i][label] - np.dot(self.prob[i], self.fm[i])
    return sum_f - self.c * np.sum(self.w)

  def _negative_log_likelihood(self):
    return np.sum(np.log(self.prob[np.arange(len(self.labels)), self.labels])) \
                  -(self.c/2)*np.linalg.norm(self.w)


def main():
  docs = ["good bad boring good",
          "exciting exciting",
          "bad boring boring boring",
          "bad exciting bad"]
  labels = [0, 0, 1, 1]

  llm = LLM(eta=0.1, c=0.1, stop_num=1.0e-5)
  llm.fit(docs, labels)

  print(llm.predict(["bad exciting good", "bad boring"]))

if __name__ == "__main__":
  main()
