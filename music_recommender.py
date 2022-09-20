import pandas as pd
from sklearn.tree import DecisionTreeClassifier

a = int(input("your age: "))
b = int(input("your gender 1 male and 0 female: "))
music_data = pd.read_csv("../Music_recommender/music.csv")
X = music_data.drop(columns=["genre"])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)
pridictions = model.predict([[a, b]])
print(pridictions)
