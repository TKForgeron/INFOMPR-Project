import preprocess as pp
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
x_train, x_val, x_test, t_train, t_val, t_test = pp.get_train_validation_test_set()
d = np.zeros((len(t_val[0][0]),len(t_val[1][0])))

for i in range(0, len(t_val[0])):
        d [np.argmax(t_val[0], axis = 1)[i]][np.argmax(t_val[1], axis = 1)[i]] += 1

for i in range(0, len(t_train[0])):
        d [np.argmax(t_train[0], axis = 1)[i]][np.argmax(t_train[1], axis = 1)[i]] += 1

for i in range(0, len(t_test[0])):
        d [np.argmax(t_test[0], axis = 1)[i]][np.argmax(t_test[1], axis = 1)[i]] += 1
