import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/shichimi_train_da.csv")
scores = data["score"]

plt.figure()
plt.hist(scores)
plt.savefig("shichimi_scores_hist.png")