import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import os

lis=os.listdir("graphs")
lis.sort(key=len)
train=pd.DataFrame()
test=pd.DataFrame()
for file in lis:
	with open('graphs/'+file) as f:
		data=json.load(f)
	train=train.append(data['Training Loss'])
	test=test.append(data['Test Loss'])
plt.plot(test[1],test[2],label="Testing Loss")
plt.plot(train[1],train[2],label="Training Loss")
plt.ylabel("Loss")
plt.xlabel("iterations")
plt.legend(loc='upper left')
plt.show()