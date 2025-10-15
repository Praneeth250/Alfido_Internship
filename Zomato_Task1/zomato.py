import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("zomato.csv", encoding='latin-1')
df.head()

df.isnull().sum()

df = df.dropna()
df = df.drop_duplicates()

df = df[['name', 'rate', 'votes', 'approx_cost(for two people)', 'location', 'listed_in(type)', 'online_order', 'book_table']]

df['rate'] = df['rate'].apply(lambda x: str(x).replace('/5', '')).astype(float)

df['approx_cost(for two people)'] = df['approx_cost(for two people)'].apply(lambda x: str(x).replace(',', '')).astype(float)

sns.countplot(x='online_order', data=df, palette='Set2')
plt.title("Restaurants Offering Online Orders")
plt.show()

sns.countplot(x='book_table', data=df, palette='Set1')
plt.title("Restaurants Offering Table Booking")
plt.show()

df['listed_in(type)'].value_counts().head(10).plot(kind='bar', color='teal')
plt.title("Top 10 Restaurant Types")
plt.ylabel("Count")
plt.show()

sns.scatterplot(x='approx_cost(for two people)', y='rate', data=df)
plt.title("Cost vs Rating")
plt.show()