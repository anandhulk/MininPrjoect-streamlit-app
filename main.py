import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import math

q=st.title("Insincere Question Classification")

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load_model():
    with st.spinner("loading model...."):
        gloveModel=tf.keras.models.load_model("./model_v3")
        return gloveModel

gloveModel=load_model()

with open('./tokenizer3.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len=100
threshold=0.45

def main():

    def plot_pie(red,green):
        labels=["InSincere","Sincere"]
        sizes=[red,green]   
        exp=(0,0.07)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.axis('equal')
        ax.pie(sizes, labels = labels,autopct='%1.2f%%',shadow=True,colors=["#cc0000","#2db300"],explode=exp,textprops={'color':"w"})
        fig.set_facecolor("#000000")
        st.pyplot(fig)
     
    q=st.text_input("enter the question")

    def predict():
        questions=[]
        questions.append(q)
        question=tokenizer.texts_to_sequences(questions)
        question=pad_sequences(question,maxlen=max_len,padding='pre')

        with st.spinner("Predicting...."):
            pred=gloveModel.predict(question)

            if(pred[0][0])>threshold:
                result="Insincere"
                score=pred[0][0]
                print(result," ",score)
                return {0:result,1:score}
            else:
                result="Sincere"
                score=1-pred[0][0]
                print(result," ",score)
                return {0:result,1:score}

    if st.button("Submit"):
        res=predict()
        a=res[1]
        formatted_string = "{:.2f}".format(a)
        float_value = float(formatted_string)

        if(res[0]=="Sincere"):
            plot_pie(1-a,a)
            st.success('This is a Sincere Question!')
            st.write("Confidence is ",float_value*100,"%")
        else:
            plot_pie(a,1-a)
            st.warning('This is a Insinsere Question')
            st.write("Confidence is ", float_value*100, "%")  

    if st.button("About"):
        st.subheader("The model was trained on Keras LSTM layers")   
        

if __name__=="__main__":
    main()