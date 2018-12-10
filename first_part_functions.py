
# coding: utf-8

# In[ ]:

#import libraries
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

# coding: utf-8

# In[1]:

#This class create Point objects in bidimensional dimension.
class Point(object):
    '''Creates a point on a coordinate plane with values x and y.'''

    COUNT = 0

    def __init__(self, x, y):
        '''Defines x and y variables'''
        self.X = x
        self.Y = y

    def move(self, dx, dy):
        '''Determines where x and y move'''
        self.X = self.X + dx
        self.Y = self.Y + dy

    def __str__(self):
        return "Point(%s,%s)"%(self.X, self.Y) 

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def distance(self, other):
        dx = self.X - other.X
        dy = self.Y - other.Y
        return math.sqrt(dx**2 + dy**2)
    
    def line(self, other, point):
        x - self.X

#this class create line objects. A line is created between two Points.        
class Line:
    def __init__(self, p1, p2):
        self.x1=p1.X
        self.x2=p2.X
        self.y1=p1.Y
        self.y2=p2.Y
        self.a = 1 /(self.x2 - self.x1)
        self.b = - 1 / (self.y2 - self.y1)
        self.c = (self.y1 / (self.y2 - self.y1)) - (self.x1 / (self.x2 - self.x1))
        
    
    def distance(self, point):
        x0 = point.X
        y0 = point.Y
        dist = abs(self.a * x0 + self.b*y0 + self.c) / math.sqrt(self.a**2 + self.b**2)
        return dist

    def slope(self):
        return - self.a / self.b

    def perpendicular(self, point):
        x0 = point.X
        y0 = point.Y
        new_slope = self.b / self.a
        return new_slope

#CLEAN the dataframe with all the announcements.
def data_cleaning_setting(df):
    #df: dataframe with all the announcements
    #RETURN the cleaned dataframe
    
    df1=df.dropna()
    df = df1[((df1.piano != 'A') & (df1.piano != 'R') & (df1.piano != 'T') & (df1.piano != 'S') & (df1.price != 'Prezzo') & (df1.price != 'da'))]
    df['price'] = list(map(lambda x: x.replace('.',''), df.price))
    df["title+description"]=df['title']+df['description']
    return df

#CREATION of the information matrix from the dataframe 
def information_matrix(df):
    #df: dataframe with informations
    #RETURN the matrix
    
    interest_columns=["price","locali","superficie","bagni","piano"]
    a=df.as_matrix(columns=interest_columns)
    return np.matrix(a).astype(float)

#ADD all the words in final in the vocabulary (if not present)
def vocabularization(vocabulary, final, index):
    #vocabulary: dictionary with all the words 
    #final: list of strings
    #index: first index to give to the first words
    
    for word in final:
        if not(word in vocabulary):
            vocabulary[word] = str(index)
            index = index + 1
    return(vocabulary, index)


#CREATION of a vocabulary of words from the dataframe. It is saved in the file 'vocabulary_prueba.txt'
def vocabulary_creation(df):
    #df: dataframe with the descriptions and titles of the announcements
    
    c = []
    #object to generate tokens(in this case words) and punctuation is removed
    tokenizer = RegexpTokenizer(r'\w+') 
    #object to stem to tokens and lowercase words
    ps = PorterStemmer()
    #set with stopwords in italian
    stop_words = set(stopwords.words('italian')) 
    for i in list(df["title+description"][0:len(df)]):
        i=i.lower()
        words=word_tokenize(i)
        words_witout_stop_words = ["" if word in stop_words else word for word in words]
        new_words = " ".join(words_witout_stop_words).strip()
        b = tokenizer.tokenize(new_words)

        for word in b:
        #loop to stem and lowercase all the words  
            c.append(ps.stem(word))
    vocabulary= {}
    index = 0

    # IF  word not in vocabulary -> add the word
    vocabulary, index = vocabularization(vocabulary, c, index)

    op = open('vocabulary_prueba.txt', 'w', encoding="utf-8")
    op.write(json.dumps(vocabulary))
    op.close()
    
#CREATION of the inverted index given a vocabulary. It is saved in the file 'inverted_index_prueba.txt'  
def inverted_index_creation(df,vocabulary):
    #df: dataframe with titles and descriptions of the announcements
    #vocabulary: dictionary of words mapped to integers
    
    inverted_index = {}
    n=0
    c = []
    #object to generate tokens(in this case words) and punctuation is removed
    tokenizer = RegexpTokenizer(r'\w+') 
    #object to stem to tokens and lowercase words
    ps = PorterStemmer()
    #set with stopwords in italian
    stop_words = set(stopwords.words('italian'))
    for i in list(df["title+description"][0:len(df)]):
        c=[]
        i=i.lower()
        words=word_tokenize(i)
        words_witout_stop_words = ["" if word in stop_words else word for word in words]
        new_words = " ".join(words_witout_stop_words).strip()
        b = tokenizer.tokenize(new_words)
        for word in b:
            c.append(ps.stem(word))

        # CREATING INVERTED INDEX
        for word in c:
            index = vocabulary[word]
            if not (index in inverted_index):
                inverted_index[index] = ['doc_' + str(n)]
            elif not('doc_' + str(n) in inverted_index[index]):
                inverted_index[index] = inverted_index[index] + ['doc_' + str(n)]
        n+=1
    op = open(r'inverted_index_prueba.txt', 'w', encoding="utf-8")
    op.write(json.dumps(inverted_index))
    op.close()

    
#CREATION of the inverted index with tf-idf given a vocabulary. It is saved in the file 'inverted_index_2_prueba.txt'      
def inverted_index_2_creation(df,vocabulary,inverted_index):
    #df: dataframe with titles and descriptions of the announcements
    #vocabulary: dictionary of words mapped to integers
    #inverted_index: inverted index with integer and relative announcements
    
    c = []
    #object to generate tokens(in this case words) and punctuation is removed
    tokenizer = RegexpTokenizer(r'\w+') 
    #object to stem to tokens and lowercase words
    ps = PorterStemmer()
    #set with stopwords in italian
    stop_words = set(stopwords.words('italian'))
    inverted_index_2 = {}
    n=0
    for i in list(df["title+description"][0:len(df)]):
        c=[]
        i=i.lower()
        words=word_tokenize(i)
        #list with the tokens with no stop words
        words_witout_stop_words = ["" if word in stop_words else word for word in words]
        #put togeher the words_without_stops_words
        new_words = " ".join(words_witout_stop_words).strip()
        #remove puntuaction and lowercase tokens
        b = tokenizer.tokenize(new_words)
        for word in b:
        #loop to stem and lowercase all the words  
            c.append(ps.stem(word))

        # CREATING INVERTED INDEX with tf-idf values
        for word in c:
            index = vocabulary[word]

            tf = c.count(word) / len(c)
            idf = math.log(len(df)/ len(inverted_index[vocabulary[word]]))
            tf_idf = tf*idf

            if not (index in inverted_index_2):
                inverted_index_2[index] = [('doc_' + str(n), tf_idf )]
            elif not(('doc_' + str(n), tf_idf)  in inverted_index_2[index]):
                inverted_index_2[index] = inverted_index_2[index] + [('doc_' + str(n), tf_idf)]

        n+=1
    op = open(r'inverted_index_2_prueba.txt', 'w', encoding="utf-8")
    op.write(json.dumps(inverted_index_2))
    op.close()

#CREATION of the matrix from descriptions. 
def description_matrix(df,vocabulary,inverted_index_2):
    #df: dataframe with titles and descriptions of the announcements
    #vocabulary: dictionary of words mapped to integers
    #inverted_index_2: inverted index with integer and relative tf-idf and documents
    
    c = []
    tokenizer = RegexpTokenizer(r'\w+') 
    ps = PorterStemmer()
    #set with stopwords in italian
    stop_words = set(stopwords.words('italian'))
    matrix_2=np.matrix(np.zeros((len(df),len(vocabulary))))
    n=0
    for i in list(df["title+description"][0:len(df)]):
        c=[]
        i=i.lower()
        words=word_tokenize(i)
        words_witout_stop_words = ["" if word in stop_words else word for word in words]#list with the tokens with no stop words
        new_words = " ".join(words_witout_stop_words).strip()#put togeher the words_without_stops_words
        b = tokenizer.tokenize(new_words)#remove puntuaction and lowercase tokens
        for word in b:
            c.append(ps.stem(word))

        # CREATING INVERTED INDEX with tf-idf values
        for word in c:
             for j in inverted_index_2[vocabulary[word]]:
                    if j[0]=="doc_"+str(n):
                        matrix_2[n,int(vocabulary[word])-1]=j[1]
                        break
        n+=1
    return matrix_2


#APPLY the elbow method to the matrix in input. Plot the elbow curve
def cluster_process(matrix):
    #matrix: the matrix on which the elbow method is applied
    #RETURN the best number of cluster picked
    
    Sum_of_squared_distances = []
    K = range(1,30)
    for k in K:
        km = sk.KMeans(n_clusters=k) 
        km = km.fit(matrix)
        Sum_of_squared_distances.append(km.inertia_)

    distances = {}

    p1 = Point(1,Sum_of_squared_distances[0])
    p2 = Point(29,Sum_of_squared_distances[28])
    line = Line(p1,p2)

    for i in range(0,29):
        p = Point(i+1,Sum_of_squared_distances[i])
        distances[i] = (line.distance(p))

    y = Sum_of_squared_distances[list(distances.keys())[list(distances.values()).index(max(distances.values()))]]
    x = 0
    for i in range(len(Sum_of_squared_distances)):
        if Sum_of_squared_distances[i] == Sum_of_squared_distances[list(distances.keys())[list(distances.values()).index(max(distances.values()))]]:
            x = i
            break
            
    z = np.linspace(0, 30, 1000)
    plt.plot(K, Sum_of_squared_distances, 'bx-') 
    plt.xlabel('k')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    point = Point(x,y)
    print("According to the Elbow-method, the optimal number of clusters is "+str(point.X)+".")
    return point.X

#APPLY kmeans method on the matrix in input with number = k clusters
def create_save_clusters(matrix,k):
    #matrix: matrix on which the kmeans is applied
    #k = number of clusters
    
    p = sk.KMeans(n_clusters=k).fit(matrix)
    save=pd.DataFrame(p.labels_)
    save.to_csv("k"+str(k)+".csv",sep=",",index=False,header=False)

    
#CALCULATE jaccard simmilarity between two dataframes   
def jaccard_similarity_score(A, B):
    #A: dataframe
    #B: dataframe
    
    s1 = set(A)
    s2 = set(B)
    return len(s1.intersection(s2)) / len(s1.union(s2))

#APPLY jaccard similarity for each combination of the two dataframes m and n
def jaccard_similarities_combination(m,n,number_of_couples=3):
    #A: dataframe
    #B: dataframe
    
    a={}
    jaccard_scores={}
    values_m=list(m[0].unique())
    values_n=list(n[0].unique())
    for i in (values_m):
        for j in (values_n):
            B=m[m[0]==i]
            A=n[n[0]==j]
            A_1=np.array(A.index)
            B_1=np.array(B.index)
            jaccard_scores[str(i)+","+str(j)]=jaccard_similarity_score(A_1,B_1)
    ik=sorted(list(jaccard_scores.values()), reverse=True)
    tbest=pd.Series(ik).head(3)

    for i in tbest:
         a[(list(jaccard_scores.keys())[list(jaccard_scores.values()).index(i)])]=jaccard_scores[(list(jaccard_scores.keys())[list(jaccard_scores.values()).index(i)])]
            
    print('The three most similar couples of cluster are:')
    print('Cluster '+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[0])])).split(",")[0])+"(information matrix) and cluster "+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[0])])).split(",")[1])+"(description matrix)  with a Jaccard similarity of "+str(sorted(a.values(),reverse=True)[0])+".")
    print('Cluster '+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[1])])).split(",")[0])+"(information matrix) and cluster "+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[1])])).split(",")[1])+"(description matrix)  with a Jaccard similarity of "+str(sorted(a.values(),reverse=True)[1])+".")
    print('Cluster '+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[2])])).split(",")[0])+"(information matrix) and cluster "+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[2])])).split(",")[1])+"(description matrix)  with a Jaccard similarity of "+str(sorted(a.values(),reverse=True)[2])+".")
    return a


#PRINT the numbers observation of the three most similar couples of cluster
def cluster_dim(a,n,s):
    #a: dictionary whose keys are the indexes of a couple of clusters and stores the values of the their cosine similarity
    #n: dataframe
    #s: dataframe
    b=[int((((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[0])])).split(",")[0])),
       int((((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[1])])).split(",")[0])),
       int((((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[2])])).split(",")[0]))]
    c=[int((((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[0])])).split(",")[1])),
       int((((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[1])])).split(",")[1])),
       int((((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[2])])).split(",")[1]))]

    print("There are "+ str(len(n[n[0]==b[0]]))+" observations in the cluster "+str(b[0])+" (information matrix)  and "+str(len(s[s[0]==c[0]]))+" observations in the cluster "+str(c[0])+"(description matrix).")
    print("There are "+ str(len(n[n[0]==b[1]]))+" observations in the cluster "+str(b[1])+" (information matrix)  and "+str(len(s[s[0]==c[1]]))+" observations in the cluster "+str(c[1])+"(description matrix).")
    print("There are "+ str(len(n[n[0]==b[2]]))+" observations in the cluster "+str(b[2])+" (information matrix)  and "+str(len(s[s[0]==c[2]]))+" observations in the cluster "+str(c[2])+"(description matrix).")
    return [b,c]
#CREATION of a color function used to draw words. 
def color_func(word,font_size,position,orientation,random_state=None,**kwargs):
    h = 0;
    s = math.floor(random.randint(30, 100));
    l = math.floor(random.randint(30, 100));
    color = 'hsl(' + str(h) + ', ' + str(s) + '%, ' + str(l) + '%)';
    return color
        
def cluster_comparison_1(variable_1,variable_2):
    #A: dataframe
    #B: dataframe
    
    a={}
    jaccard_scores={}
    values_variable_1=list(variable_1[0].unique())
    values_variable_2=list(variable_2[0].unique())
    for i in (values_variable_1):
        for j in (values_variable_2):
            B=variable_1[variable_1[0]==i]
            A=variable_2[variable_2[0]==j]
            A_1=np.array(A.index)
            B_1=np.array(B.index)
            jaccard_scores[str(i)+","+str(j)]=jaccard_similarity_score(A_1,B_1)
    ik=sorted(list(jaccard_scores.values()), reverse=True)
    tbest=pd.Series(ik).head(4)
    for i in tbest:
         a[(list(jaccard_scores.keys())[list(jaccard_scores.values()).index(i)])]=jaccard_scores[(list(jaccard_scores.keys())[list(jaccard_scores.values()).index(i)])]
            
    print('Comparisons of clusters:')
    print('Cluster '+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[0])])).split(",")[0])+"(KMeans++) and cluster "+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[0])])).split(",")[1])+"(KMeans)  with a Jaccard similarity of "+str(sorted(a.values(),reverse=True)[0])+".")
    print('Cluster '+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[1])])).split(",")[0])+"(KMeans++) and cluster "+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[1])])).split(",")[1])+"(KMeans)  with a Jaccard similarity of "+str(sorted(a.values(),reverse=True)[1])+".")
    print('Cluster '+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[2])])).split(",")[0])+"(KMeans++) and cluster "+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[2])])).split(",")[1])+"(KMeans)  with a Jaccard similarity of "+str(sorted(a.values(),reverse=True)[2])+".")
    print('Cluster '+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[3])])).split(",")[0])+"(KMeans++) and cluster "+str(((list(a.keys())[list(a.values()).index(sorted(a.values(),reverse=True)[3])])).split(",")[1])+"(KMeans)  with a Jaccard similarity of "+str(sorted(a.values(),reverse=True)[3])+".")
    return a

    
#GENERATION of a wordcloud and fill an image with words
def wordcloud_generator(n,m,n_1,m_1,image,df):
    #n, m : dataframes with clusters
    #n_1, m_1: number of the cluster
    #image: image to be filled with word(must be only in white and black)
    #df: dataframe
    txt =''
    d = ''
    c = []
    tokenizer = RegexpTokenizer(r'\w+') #object to generate tokens(in this case words) and punctuation is removed
    ps = PorterStemmer() #object to stem to tokens and lowercase words
    stop_words = set(stopwords.words('italian'))
    c=[]
    txt=txt.lower()
    for i in set(n[n[0]==n_1].index).union(set(m[m[0]==m_1])):
        txt =  str((list(df[i:i+1]["description"])))
        for i in list([txt]):
            c=[]
            i=i.lower()
            words=word_tokenize(i)
            words_witout_stop_words = ["" if word in stop_words else word for word in words] #list with the tokens with no stop words
            new_words = " ".join(words_witout_stop_words).strip() #put togeher the words_without_stops_words
            b = tokenizer.tokenize(new_words) #remove puntuaction and lowercase tokens
        d+=" ".join(b)

    e=wordcloud.WordCloud(mask=image,max_words=100,min_font_size=2, background_color="white",contour_color=(21,109,143),contour_width=3).generate((d))
    plt.figure(figsize=[8,8])
    plt.imshow(e.recolor(color_func=color_func, random_state=3),interpolation="bilinear")
    plt.axis("off")
    plt.show()


