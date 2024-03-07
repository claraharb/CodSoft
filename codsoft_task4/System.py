import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# data
data = pd.read_csv(r"C:\Users\Admin\Desktop\spotify.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())


df = data.drop(columns=['id', 'name', 'artists', 'release_date', 'year'])

print(df.corr())


datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
normarization = data.select_dtypes(include=datatypes)
print(normarization)


scaler = MinMaxScaler()
normarization_normalized = scaler.fit_transform(normarization)
normarization_normalized = pd.DataFrame(normarization_normalized, columns=normarization.columns)
print(normarization_normalized)


kmeans = KMeans(n_clusters=10)
features = kmeans.fit_predict(normarization_normalized)
data['features'] = features
print(data['features'])

# Select a song
song_title = "I Don't Wanna Be Kissed"
distance = []


song = data[data['name'].str.lower() == song_title.lower()].head(1).values[0]
rec = data[data['name'].str.lower() != song_title.lower()]


for s in tqdm(rec.values):
    d = 0
    for col in np.arange(len(rec.columns)):
        if col not in [1, 6, 12, 14, 18]:
            d += np.absolute(float(song[col]) - float(s[col]))
    distance.append(d)

rec['distance'] = distance
rec = rec.sort_values('distance')
columns = ['artists', 'name']
print(rec[columns][:5])
