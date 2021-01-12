"""IMPORT EDİLECEK KÜTÜPHANELER
json
numpy
keras
tensorflow
pickle
nltk
snowballstemmer
"""
import json #datasetlerimiz json metin dosyası formatında olacak.
import numpy as np
import random
import pickle #modeller pickle dosyası şeklinde kaydedilecek.
from tensorflow.keras.models import Sequential #modellerimizdeki katmanların lineer bir dizisini tutacağız.
from tensorflow.keras.layers import Dense, Embedding, Dropout, Activation, GlobalAveragePooling1D #katmanlarımız için gerekli olan yapılar.
from tensorflow.keras.optimizers import SGD #gradient descent optimizasyonları için kullanacağız.
import nltk #dil işleme kütüphanemiz.
from snowballstemmer import TurkishStemmer #türkçe destekle kelime köklerini ayıracağız.

nltk.download("punkt") #cümleleri kelimelere aıyrmak için öncelikle nltk modülümüzü indiriyoruz.

with open("dataset.json") as file: #dataset dosyamızı açıyoruz.
    intents=json.load(file) #data değişkenine json dosyası açıldı.

stemmer=TurkishStemmer() #kök ayırma işlemini türkçe destekle yapıyoruz.

words=[] #ayıklanmış kelimelerimizin tutulacağı liste.
classes=[] #json dosyamızdaki etiketlerimizin tutulacağı liste.
documents=[] #json dosyamızdaki etiket ve patternların tutulacağı liste.
ignore_letters=["!","'","?",",","."] #cümle içindeki bu noktalama işaretlerini atlıyoruz.

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word=nltk.word_tokenize(pattern) #json dosyamızdaki patternlerdeki cümleleri kelimelere ayırıyoruz.
        words.extend(word) #ayırdığımız kelimeleri listeye ekliyoruz.
        print(words)
        documents.append((word, intent["tag"])) #ayıklanmış kelime listemizi ve ait olduğu etiketi ekliyoruz.
        if intent["tag"] not in classes:
            classes.append(intent["tag"]) #etiketimizi listeye ekliyoruz.

words=[stemmer.stemWord(w.lower()) for w in words if w not in ignore_letters] #kelimelerin köklerini alma,
#harfleri küçültme ve atlayacağımız işaretleri kontrol etme işlemlerini yapıp kelimelerimizi düzenliyoruz.
words=sorted(list(set(words))) #kelimelerimizi kümeye çevirip sıralıyoruz. küme olduğu için aynı elemanlar tekrar sayılmadı.
classes=sorted(list(set(classes))) #aynı işlemi etiketlerimiz için yapıyoruz.
print(len(documents), "adet etiketli girdi var.")
print(len(classes), "adet etiket var.")
print(len(words), "adet ayıklanmış benzersiz kelime var.")

pickle.dump(words, open("words.pkl", "wb")) #kelimelerimizi arayüzde kullanabilmek için binary formda pickle dosyalarına kaydediyoruz.
pickle.dump(classes, open("classes.pkl", "wb")) #aynı şekilde etiketlerimizi de kaydediyoruz.

training_data=[]
output_empty=[0]*len(classes) #boş bir output listemiz. etiketlerimizin uzunluğu kadar bir uzunluğu olacak.

for doc in documents: #eğitim setlerimizi oluşturacağız.
    bag=[] #kelimelerimizi 0 ve 1 değerlerine çevirip öğrenmeyi sağyalayacağız.
    pattern_words=doc[0] #ayıklanmış kelimelerimizi alıyoruz.
    pattern_words=[stemmer.stemWord(word.lower()) for word in pattern_words] #kelimelerimizi ayıklıyoruz.
    for word in words:
        if word in pattern_words: #eğer en başta ayıkladığımız kelime eğitim seti için ayıklanmış kelimeler
            #içinde varsa 1, eğer yoksa 0 ekliyoruz.
            bag.append(1)
        else:
            bag.append(0)


    output_row=list(output_empty) #eğitim sonucu çıkış değerimiz.
    output_row[classes.index(doc[1])]=1 #lineer hale getirdiğimiz kelimenin etiketini de 1 yapıyoruz, diğer etiketler 0.

    training_data.append([bag, output_row]) #eğitim datamıza hem ayıklanmış kelimelerimizin hem de etiketlerimizin
    #lineer hale çevrilmiş 0-1 halini ekliyoruz. modelimizi bu datayı kullanarak eğiteceğiz.
    print("training_data: ",len(training_data), training_data)

random.shuffle(training_data) #eğitim datamızı daha iyi eğitebilmek için karıştırıyoruz.
training_data=np.array(training_data) #eğitim datamızı array'e çeviriyoruz.
train_x=list(training_data[:,0]) #training datamızın bütün 0. indislerini train_x listemize atıyoruz. patternler
train_y=list(training_data[:,1]) #train datamızın bütün 1. indislerini train_y listemize atıyoruz. etiketler
print("train data created.")

model=Sequential() #katmanlı modelimiz.
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu")) #giriş katmanı 128 nöron içeriyor ve fonksiyonu relu.
model.add(Dropout(0.5)) #ezberi önlemek için dropout değerimiz 0.5.
model.add(Dense(64, activation="relu")) #ikinci katmanımız 64 nöron ve fonksiyonu relu.
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax")) #çıkış katmanımız etiket sayısı kadar nörona sahip ve fonksiyonu softmax.

sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #modelin optimizasyonu için SGD kullanıyoruz.
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]) #modeli compile ediyoruz.

fittedmodel=model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) #modelimizin formu.
model.save("chatbot.h5", fittedmodel) #modelimizi kaydediyoruz.

print("model created.")

