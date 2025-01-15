import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time

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

def get_article_text(url):

    to_save = []

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        classes_to_check = [
            "caas-body",
            "col-body",
            "article-paywall-contents",
            "paywall"
        ]

        text_body = None
        used_class = None

        for class_name in classes_to_check:
            text_body = soup.find("div", class_=class_name)
            if text_body:
                used_class = class_name
                break

        if not text_body:ANO
            print(f"No content found for URL: {url}")
            return None

        print(f"Content found for URL: {url}, using class: {used_class}")

        article = [el.text for el in text_body.find_all('p')]
        print(article, "\n")

        to_save.append([url, article])
        new_df = pd.DataFrame(to_save, columns=['Link', 'Text'])
        new_df.to_csv(f"{url}")

        return ' '.join(article)

    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None
