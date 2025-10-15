import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
files = {
    'comments': 'comments.csv',
    'follows': 'follows.csv',
    'likes': 'likes.csv',
    'photos': 'photos.csv',
    'users': 'users.csv'
}

sns.set_style("whitegrid")
print("Starting Simple Instagram Data Analysis...")
dataframes = {}
for name, filepath in files.items():
    try:
        df = pd.read_csv(filepath)
        dataframes[name] = df
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Skipping.")
        exit()
if 'users' in dataframes:
    dataframes['users'].rename(columns={'post count': 'post_count'}, inplace=True)

if 'photos' in dataframes:
    dataframes['photos'].rename(columns={'user ID': 'user_id', 'photo type': 'content_type'}, inplace=True)

if 'likes' in dataframes:
    dataframes['likes'].rename(columns={'user': 'user_id', 'photo': 'photo_id'}, inplace=True)

if 'comments' in dataframes:
    dataframes['comments'].rename(columns={'Photo id': 'photo_id'}, inplace=True)
if 'photos' in dataframes:
    photos_df = dataframes['photos']
    plt.figure(figsize=(8, 6))
    content_counts = photos_df['content_type'].str.lower().value_counts()
    sns.barplot(x=content_counts.index, y=content_counts.values, palette='viridis')
    plt.title('1. Distribution of Content Types')
    plt.xlabel('Content Type')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()
if all(df in dataframes for df in ['likes', 'photos']):
    likes_df = dataframes['likes']

    photo_likes = likes_df.groupby('photo_id')['user_id'].count().reset_index(name='total_likes')

    top_liked_photos = photo_likes.sort_values(by='total_likes', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='total_likes', y='photo_id', data=top_liked_photos.astype({'photo_id': 'str'}), palette='flare')
    plt.title('2. Top 10 Most Liked Photos')
    plt.xlabel('Total Likes')
    plt.ylabel('Photo ID')
    plt.tight_layout()
    plt.show()
    plt.close()

if all(df in dataframes for df in ['follows', 'users']):
    follows_df = dataframes['follows']
    follower_counts = follows_df['followee'].value_counts().reset_index(name='follower_count')
    follower_counts.rename(columns={'index': 'user_id'}, inplace=True)

    top_followed_users = follower_counts.merge(dataframes['users'][['id', 'name']], left_on='user_id', right_on='id', how='left')
    top_followed_users.sort_values(by='follower_count', ascending=False, inplace=True)
    top_followed = top_followed_users.head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='follower_count', y='name', data=top_followed, palette='rocket')
    plt.title('3. Top 10 Most Followed Users')
    plt.xlabel('Follower Count')
    plt.ylabel('User Name (Followee)')
    plt.tight_layout()
    plt.show()
    plt.close()

print("\nSimple analysis complete! The three visualizations should now be displayed.")