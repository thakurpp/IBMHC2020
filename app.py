import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt

def main():
    st.title("Wind Energy Forcaster Web App")
    st.sidebar.title("Wind Energy Forcaster")
    st.markdown("KNOW THE ENERGY PRODUCTION OF YOUR POWER PLANT⚡")


    @st.cache(persist=True)
    def load_data():
        data=pd.read_csv('C:/Users/palash/Downloads/133415_317642_bundle_archive/T1.csv')
        data=data[['LV ActivePower (kW)','Wind Speed (m/s)','Wind Direction (°)']]
        
        return data
    df=load_data()    

    @st.cache(persist=True)
    def split(df):
        train_dataset = df.sample(frac=0.8,random_state=0)
        test_dataset = df.drop(train_dataset.index)

        train_stats = train_dataset.describe()
        train_stats.pop("LV ActivePower (kW)")
        train_stats = train_stats.transpose()
        train_labels = train_dataset.pop('LV ActivePower (kW)')
        test_labels = test_dataset.pop('LV ActivePower (kW)')


        def norm(x):


             return (x - train_stats['mean']) / train_stats['std']
        normed_train_data = norm(train_dataset)
        normed_test_data = norm(test_dataset)   

        return normed_train_data,normed_test_data,train_labels,test_labels

    normed_train_data,normed_test_data,train_labels,test_labels=split(df)

    if st.sidebar.checkbox("Show Input Data",False):
        st.subheader("Wind Energy Data Set")
        st.markdown("Source-This file was taken from a wind turbine's scada system that is working and generating power in Turkey. The coordinates are X:668478 Y:4494833 UTM ED 50, 6 degree")
        st.write(df)

    st.sidebar.subheader("Enter the Input")
    st.sidebar.markdown("(Enter both values in Hourly Interval)")
    Wind_Speed=st.sidebar.number_input("Wind Speed (m/s)💨",1,200,step=1) 
    Wind_Direction = st.sidebar.number_input("Wind Direction (°)🧭",1,360,step=1)
    st.markdown("Your input data in tabular format can be viewed below")
    @st.cache(allow_output_mutation=True)
    def get_data():
        return []

    #Wind_Speed = st.sidebar.number_input("Wind Speed")
    #Wind_Direction = st.sidebar.number_input("Wind Direction")
    if st.button("Add row"):
        get_data().append({"Wind Speed (m/s)": Wind_Speed, "Wind Direction (°)": Wind_Direction})

    st.write(pd.DataFrame(get_data()))

    test_data=pd.DataFrame(get_data())
    #st.write(pd.DataFrame(test_data[['Wind Speed','Wind Direction']]))
    
    def data_pre():
        train_dataset = df.sample(frac=0.8,random_state=0)
        test_dataset = df.drop(train_dataset.index)

        train_stats = train_dataset.describe()
        train_stats.pop("LV ActivePower (kW)")
        train_stats = train_stats.transpose()
        train_labels = train_dataset.pop('LV ActivePower (kW)')
        test_labels = test_dataset.pop('LV ActivePower (kW)')


        def norm(x):


             return (x - train_stats['mean']) / train_stats['std']
        normed_train_data = norm(train_dataset)
        normed_test_data = norm(test_dataset)   
        out=norm(test_data[['Wind Speed (m/s)','Wind Direction (°)']])

        return out

    

    
    knn = KNeighborsRegressor(n_neighbors=60, metric = 'minkowski', p = 3).fit(normed_train_data, train_labels)

    plot=st.sidebar.multiselect("Select Graph type to be Viewed📈",('Scatter Plot','Line Plot','Power Curve','View All')) 
    

    def p(plot_list):
        if 'Line Plot' in plot_list:
            st.subheader("Line Plot Based On Your Data")
            plt.plot(output.index,output['Power Generated(KW)'])
            plt.xlabel('Time(Each Unit Indicates 1 Hr Interval)')
            plt.ylabel('Power Output(KW)')
            st.pyplot()
        if 'Scatter Plot' in plot_list:
            st.subheader("Scatter Plot Based On Your Data")
            plt.scatter(output.index,output['Power Generated(KW)'])
            plt.xlabel('Time(Each Unit Indicates 1 Hr Interval)')
            plt.ylabel('Power Output(KW)')
            st.pyplot()
        if 'View All' in plot:
            st.subheader("Line Plot Based On Your Data")
            plt.plot(output.index,output['Power Generated(KW)'])
            plt.xlabel('Time(Each Unit Indicates 1 Hr Interval)')
            plt.ylabel('Power Output(KW)')
            st.pyplot()
            st.subheader("Scatter Plot Based On Your Data")
            plt.scatter(output.index,output['Power Generated(KW)'])
            plt.xlabel('Time(Each Unit Indicates 1 Hr Interval)')
            plt.ylabel('Power Output(KW)')
            st.pyplot()
            st.subheader("Power Curve")
            plt.plot(test_data['Wind Speed (m/s)'],output['Power Generated(KW)'])
            plt.xlabel('Wind Speed (m/s)')
            plt.ylabel('Power Generated(KW)')
            st.pyplot()
        if 'Power Curve' in plot:
            st.subheader("Power Curve")
            plt.plot(test_data['Wind Speed (m/s)'],output['Power Generated(KW)'])
            plt.xlabel('Wind Speed (m/s)')
            plt.ylabel('Power Generated(KW)')
            st.pyplot()





    if st.sidebar.button("Forcast",key='Forcast'):
        st.subheader("Prediction results based on your input")
        #reg1 = GradientBoostingRegressor(random_state=1).fit(normed_train_data, train_labels)
        #forest = RandomForestRegressor(n_estimators = 60, criterion = 'mse').fit(normed_train_data, train_labels)
        #svc_rbf = SVR(kernel = 'rbf').fit(normed_train_data, train_labels)
        #knn = KNeighborsRegressor(n_neighbors=60, metric = 'minkowski', p = 3).fit(normed_train_data, train_labels)
        #vote=VotingRegressor([('rg', reg1), ('rf', forest), ('knn', knn),('svm',svc_rbf)]).fit(normed_train_data, train_labels) 
        y_pred=knn.predict(data_pre())
        output=pd.DataFrame(y_pred)
        output.rename(columns = {0:'Power Generated(KW)'}, inplace = True)
        st.write(output)
        p(plot)
        st.write("The best time to harvest maximum energy based on your data is at {}th hour from now".format(np.array(output).argmax()))

    st.markdown("Reference URL for weather data is {} \n {}".format('https://weather.com/en-IN/','https://www.weather.gov/gyx/WindSpeedAndDirection'))     






if __name__ == '__main__':
    main()


