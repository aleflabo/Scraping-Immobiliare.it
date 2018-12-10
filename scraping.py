
# coding: utf-8

# In[1]:
import pandas as pd    
import numpy as np
import nltk
import csv
import io
import sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import json
import math
import sklearn.cluster as sk
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import wordcloud
from PIL import Image
import math
import random

#CONTROL if the link in input has the correct form. some links does not have the initial URL part
def controlled(link):
    #link: string
    #RETURN the link with the missing part 
    if link[:4] != 'http':
        link = 'https://www.immobiliare.it' + link 
    return link


#FIND 10000 announcements from immobiliare.it. At the end they are stored in a file called 'data.csv' 
def scraping():
    
    #finding the announcements (more than 10000) 
    links = []
    idx = 1
    count = 0
    while (len(links)< 10000):

        url = 'https://www.immobiliare.it/vendita-case/roma/?criterio=rilevanza&pag=' + str(idx)
        idx += 1
        content = requests.get(url)
        soup = BeautifulSoup(content.text, "lxml")
        if count == 100:
            time.sleep(1)
            count = 0

        divTag = soup.find_all("div", {'class': "listing-item_body--content"})
        for tag in divTag:
            tdTags = tag.find_all("a")
            for tag in tdTags:
                l = controlled(tag['href'])
                links.append(l)


    #retrieving info for each url found before
    r = pd.DataFrame(index = [0,1,2,3,4,5,6])
    idx = 0
    count = 1
    for url in links: 
        try:
            content = requests.get(url)
            soup = BeautifulSoup(content.text, "lxml")
            row = []

            if count == 100:
                time.sleep(1)
                count = 0
            divTag = soup.find_all("h1", {'class': "raleway title-detail"})
            for title in divTag:
                row.append(title.text)

            divTag = soup.find_all("div", {'class': "clearfix description"})
            for desc in divTag:
                desc = desc.find_all("div")
                h = []
                for tag in desc:
                    h.append((tag.text.replace('\n', ' ')))

            row.append(h[1])

            divTag = soup.find_all("div", {'class': "im-property__features"})
            for tag in divTag:

                price = tag.find_all("li", {'class': 'features__price'})
                for tag_p in price:
                    row.append(((tag_p.text.replace('€', ' ')).split())[0])

                info = tag.find_all("ul", {'class': 'list-inline list-piped features__list'})
                h = []
                for tag in info:
                    for tg in tag:
                        h.append((tg.text.replace('\xa0', ' ').replace('\n', ' ').replace('+', ' ')).split())


            for i in h[:-1]:
                if i[0] == 'da':
                    row.append(i[1])
                else:
                    row.append(i[0])
            r[idx] = pd.DataFrame(row)
            idx += 1

        except:
            print(idx)
            idx += 1

    #save the dataframe in data.csv file         
    r = r.transpose()
    r.columns = ['title','description', 'price', 'locali', 'superficie', 'bagni', 'piano']
    r.to_csv('data.csv', sep = ",",header= True, index = False)

