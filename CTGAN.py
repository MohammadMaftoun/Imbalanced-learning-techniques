#CTGAN Libraries
from ctgan import CTGAN


cat_feature = ['example1','example2']

ctgan = CTGAN(verbose=True)
ctgan.fit(data, cat_feature, epochs = 200)

# Saving ctgan
ctgan.save('file.pkl')

# Creating samples
samples = ctgan.sample()

# Fixing class imbalance problem
samples = samples[samples['target'] ==1]
ctgan_result_df = pd.concat([data,samples])
ctgan_result_df.head()
