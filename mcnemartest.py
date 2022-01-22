from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
matrix = np.zeros((2,2)) 

for i in range(0, len(t_test[0])):
    a = np.where(t_test[0][i] == max(t_test[0][i]))[0][0] ==  np.where(cnn_results[0][i] == max(cnn_results[0][i]))[0][0]
    b = np.where(t_test[0][i] == max(t_test[0][i]))[0][0] ==  np.where(rnn_results[0][i] == max(rnn_results[0][i]))[0][0]
    matrix[int(not a)][int(not b)] += 1

test = mcnemar(matrix, exact=True)
print(test.pvalue)