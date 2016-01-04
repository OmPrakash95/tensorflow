import numpy as np


def LevensteinDistance(ref, hyp):
  d = np.zeros([len(hyp) + 1, len(ref) + 1], dtype=np.int64)

  for i in range(1, len(ref) + 1):
    d[0, i] = i
  for j in range(1, len(hyp) + 1):
    d[j, 0] = j

  for j in range(1, len(hyp) + 1):
    for i in range(1, len(ref) + 1):
      if ref[i - 1] == hyp[j - 1]:
        d[j, i] = d[j - 1, i - 1]
      else:
        deletion     = d[j, i - 1] + 1
        insertion    = d[j - 1, i] + 1
        substitution = d[j - 1, i - 1] + 1

        if deletion < insertion and deletion < substitution:
          d[j, i] = deletion
        elif insertion < substitution:
          d[j, i] = insertion
        else:
          d[j, i] = substitution

  return d[len(hyp), len(ref)]
