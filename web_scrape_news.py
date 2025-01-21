import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
import os
import ast


def get_news_yahoo(keywords = None, max_page = 2, save_csv = True, filename = 'news_dataset'):

    if keywords is None:
        keywords = ['Trump', 'polish', 'euro', 'german']

    news_data = []

    for keyword in keywords:
        print(keyword)
        for page in (0, max_page):

            url = 'https://news.search.yahoo.com/search?q={}&b={}'.format(keyword,page)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            for news_item in soup.find_all('div', class_='NewsArticle'):

                news_title = news_item.find('h4').text
                news_source = news_item.find('span','s-source').text
                news_description = news_item.find('p','s-desc').text
                news_link = news_item.find('a').get('href')
                news_time = news_item.find('span', class_='s-time').text

                news_time = news_time.replace('·', '').strip()
                news_title = news_title.replace('•', '').strip()

                news_data.append([news_title, news_source, news_description, news_link, news_time])

                time.sleep(random.uniform(0.5, 1))

        time.sleep(random.uniform(1, 2))

        half_way_data = pd.DataFrame(news_data, columns=['Title', 'Source', 'Description', 'Link', 'Time'])
        if save_csv:
            half_way_data.to_csv(filename)


    news_data_df = pd.DataFrame(news_data, columns=['Title','Source','Description','Link','Time'])

    if save_csv:
        news_data_df.to_csv(filename)

    return news_data_df

def get_article_text(title, url, path):

    to_save = []

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        classes_to_check = [
            "caas-body",
            "col-body",
            "article-paywall-contents",
            "paywall",
            "article-paywall-contents",
            "article-body",
            "article-body v_text",
            "article-content article-body rich-text",
            "entry__content-list",
            "articleBody",
            "sc-dbXBFb gGVbCD",
            "text-description",
            "body yf-tsvcyu"
        ]

        text_body = None
        used_class = None

        for class_name in classes_to_check:
            text_body = soup.find("div", class_=class_name)
            if text_body:
                used_class = class_name
                break

        if not text_body:
            print(f"No content found for URL: {url}")
            return None

        print(f"Content found for URL: {url}, using class: {used_class}")

        article = [el.text for el in text_body.find_all('p')]

        article = ' '.join(article)

        to_save.append([title, url, article])
        new_df = pd.DataFrame(to_save, columns=['Link', 'Text'])
        new_df.to_csv(f"{path}/{title}.csv")

        return ' '.join(article)

    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None


def get_article_by_link(df, path="articles"):


    for title, link in zip(df['Title'], df['Link']):

        if not link.startswith("https://www.washingtonpost.com"):
            res = get_article_text(title, link, path)


def create_df_articles(folder_path='articles'):

    df = {'title': [], 'link': [], 'text': []}

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            try:
                # Read the CSV file
                df2 = pd.read_csv(file_path)

                # Validate required columns exist
                if 'Link' not in df2.columns or 'Text' not in df2.columns:
                    print(f"Skipping {file_name}: Missing required columns.")
                    continue

                # Extract the title, link, and text
                title = os.path.splitext(file_name)[0]
                link = df2['Link'].tolist()  # Convert Series to list
                text = df2['Text'][0] if not df2['Text'].empty else ''

                # Parse and join the text
                try:
                    text = ast.literal_eval(text) if isinstance(text, str) else text
                    text = ' '.join(text) if isinstance(text, list) else str(text)
                except (ValueError, SyntaxError) as e:
                    print(f"Skipping text parsing in {file_name}: {e}")
                    text = str(text)  # Fallback to raw text

                # Append to the DataFrame
                df['title'].append(title)
                df['link'].append(link)
                df['text'].append(text)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    return pd.DataFrame(df)