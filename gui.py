from nltk.stem import PorterStemmer #ingilizce cümleler için.
import nltk
from snowballstemmer import TurkishStemmer
import pickle
import numpy as np
import json
import random
from tkinter import *
from tensorflow.keras.models import load_model

nltk.download("punkt")
stemmer=TurkishStemmer()
ps=PorterStemmer()

dataset1=json.loads(open("b1.json", encoding='utf8').read())
dataset2=json.loads(open("bb2.json", encoding='utf8').read())
dataset3=json.loads(open("c1.json", encoding='utf8').read())
dataset4=json.loads(open("c2.json", encoding='utf8').read())

model=load_model("chatbot.h5") #modelimizi yüklüyoruz.
dataset=json.loads(open("dataset.json").read()) #datasetimizi yüklüyoruz.
words=pickle.load(open("words.pkl", "rb")) #kelimelerimizi yüklüyoruz.
classes=pickle.load(open("classes.pkl", "rb")) #etiketlerimizi yüklüyoruz.

###############################################################################################################

def raw_sentence(sentence): #cümle düzenleyeceğiz.
    sentence_words=nltk.word_tokenize(sentence) #gelen cümleyi kelimelere ayırıyoruz.
    sentence_words=[stemmer.stemWord(word.lower()) for word in sentence_words] #kelimeleri köklere ayırıyoruz.
    return sentence_words

def words_bag(sentence, words, show_details=True): #cümle içindeki kelimelerin 0-1 karşılığını döndüreceğiz.
    sentence_words=raw_sentence(sentence)
    bag=[0]*len(words)
    for s in sentence_words:
        for i, word in enumerate(words): #kelimeleri harf harf ayırıp bir numara atıyoruz.
            if word==s:
                bag[i]=1
    return (np.array(bag))

def predict(sentence): #BURADAN HİÇBİR ŞEY ANLAMADIM
    p=words_bag(sentence, words, show_details=False)
    res=model.predict(np.array([p]))[0]
    ERROR_TRESHOLD=0.25
    results= [[i,r] for i,r in enumerate(res) if r>ERROR_TRESHOLD]
    results.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({"intent":classes[r[0]], "probability":str(r[1])})
    return return_list

def response(ints, intents_json): #cevap verme fonksiyonumuz.
    tag=ints[0]["intent"]
    intents_list=intents_json["intents"] #etiketlerimizi alıyoruz.
    for i in intents_list:
        if(i["tag"]==tag): #eğer şu anki etiketimiz listemizde varsa,
            result=random.choice(i["responses"]) #o etiketten rastgele bir cevap seçiyoruz.
            break
    return result

#################################################################################################################

def send():
    msg=entrybox.get("1.0", "end-1c").strip()
    entrybox.delete("0.0", END)

    if msg != " ":
        chatbox.config(state=NORMAL)
        chatbox.insert(END, "User: "+msg+"\n\n")
        chatbox.config(foreground="#446665", font=("Times", 12))

        entry_word=nltk.word_tokenize(msg)
        entry_word=[ps.stem(word.lower()) for word in entry_word]
        word_list1=dataset1["B1Words"]
        word_list2=dataset2["B2Words"]
        word_list3=dataset3["C1Words"]
        word_list4=dataset4["C2Words"]

################################################################################################################

        if msg=="b1" or msg=="B1" or msg=="c1" or msg=="C1" or msg=="b2" or msg=="B2" or msg=="c2" or msg=="C2":
            file=open("seviye.txt","w")
            file.write(msg)
            file.close()

            chatbox.insert(END, "Hello ChatBot: Thank you for entering your level!" + "\n\n")

            file=open("seviye.txt","r")
            seviye=file.readline()

            for i in range(5):
                if seviye=="b1" or seviye=="B1":
                    kelime=random.choice(dataset1["B1Words"])
                    sonuc=(kelime["Word"]+"\n"+kelime["Meaning"]+"\n"+kelime["Sentence"])

                    chatbox.insert(END, "Hello ChatBot: " + sonuc +"\n\n")

                if seviye=="b2" or seviye=="B2":
                    kelime=random.choice(dataset2["B2Words"])
                    sonuc=(kelime["Word"]+"\n"+kelime["Meaning"]+"\n"+kelime["Sentence"])

                    chatbox.insert(END, "Hello ChatBot: " + sonuc +"\n\n")

                if seviye=="c1" or seviye=="C1":
                    kelime=random.choice(dataset3["C1Words"])
                    sonuc=(kelime["Word"]+"\n"+kelime["Meaning"]+"\n"+kelime["Sentence"])

                    chatbox.insert(END, "Hello ChatBot: " + sonuc +"\n\n")

                if seviye=="c2" or seviye=="C2":
                    kelime=random.choice(dataset4["C2Words"])
                    sonuc=(kelime["Word"]+"\n"+kelime["Meaning"]+"\n"+kelime["Sentence"])

                    chatbox.insert(END, "Hello ChatBot: " + sonuc +"\n\n")

            chatbox.config(state=DISABLED)
            chatbox.yview(END)

            file.close()

#################################################################################################################

        for i in word_list1:
            j=nltk.word_tokenize(i["Word"])
            j=[ps.stem(word.lower()) for word in j]
            if (j==entry_word):
                result=("(B1) "+i["Meaning"]+"\n"+i["Sentence"])
                chatbox.insert(END, "Hello ChatBot: " + result + "\n\n")
                chatbox.config(state=DISABLED)
                chatbox.yview(END)
                break

        for i in word_list2:
            j=nltk.word_tokenize(i["Word"])
            j=[ps.stem(word.lower()) for word in j]
            if (j==entry_word):
                result=("(B2) "+i["Meaning"]+"\n"+i["Sentence"])
                chatbox.insert(END, "Hello ChatBot: " + result + "\n\n")
                chatbox.config(state=DISABLED)
                chatbox.yview(END)
                break

        for i in word_list3:
            j=nltk.word_tokenize(i["Word"])
            j=[ps.stem(word.lower()) for word in j]
            if (j==entry_word):
                result=("(C1) "+i["Meaning"]+"\n"+i["Sentence"])
                chatbox.insert(END, "Hello ChatBot: " + result + "\n\n")
                chatbox.config(state=DISABLED)
                chatbox.yview(END)
                break

        for i in word_list4:
            j=nltk.word_tokenize(i["Word"])
            j=[ps.stem(word.lower()) for word in j]
            if (j==entry_word):
                result=("(C2) "+i["Meaning"]+"\n"+i["Sentence"])
                chatbox.insert(END, "Hello ChatBot: " + result + "\n\n")
                chatbox.config(state=DISABLED)
                chatbox.yview(END)
                break


        ints = predict(msg)  # mesajımızı tahmin edilmesi için fonksiyona gönderdik.
        res = response(ints, dataset)  # cevap için fonksiyona gidiyoruz.
        chatbox.insert(END, "Hello ChatBot: " + res + "\n\n")
        chatbox.config(state=DISABLED)
        chatbox.yview(END)

############################################## GUI ##########################################################

pencere=Tk()
pencere.title("Hello ChatBot!")
pencere.geometry("400x500+450+100")
pencere.resizable(width=FALSE, height=FALSE)

chatbox=Text(pencere, bg="white", font="Times")

chatbox.insert(END, "Welcome to Hello ChatBot. You "
                    "can enter your level or start chatting me, or you can do some translation!"  + "\n\n")
chatbox.config(state=DISABLED)
chatbox.yview(END)

scrollbar=Scrollbar(pencere, command=chatbox.yview, cursor="star")
chatbox["yscrollcommand"]=scrollbar.set

sendbutton=Button(pencere, font=("Times", 12), text="Send", bg="orange", activebackground="grey",
                  fg="black", relief=RAISED, command=send)

entrybox=Text(pencere, bg="white", font="Times")

chatbox.place(x=10, y=10, height=380, width=350)
scrollbar.place(x=370, y=10, height=380, width=20)
entrybox.place(x=10, y=400, height=90, width=280)
sendbutton.place(x=300, y=400, height=90, width=90)

pencere.mainloop()
