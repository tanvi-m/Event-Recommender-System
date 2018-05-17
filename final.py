import numpy as np
import pandas as pd
import sklearn
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Read data from various txt files
def load_data():

    daydate = []
    file = open("daydate.txt", "r", errors='ignore') 
    for index, line in enumerate(file): 
            daydate.append(line.strip("\n"))

    time = []
    file = open("time.txt", "r", errors='ignore') 
    for index, line in enumerate(file): 
            time.append(line.strip("\n"))
        
    venue = []
    file = open("venue.txt", "r", errors='ignore') 
    for index, line in enumerate(file): 
            venue.append(line.strip("\n"))
            
    topic = []
    file = open("topic.txt", "r", errors='ignore') 
    for index, line in enumerate(file): 
            topic.append(line.strip("\n"))

    description = []
    file = open("desc.txt", "r", errors='ignore') 
    for index, line in enumerate(file): 
            description.append(line.strip("\n"))
    
    pd_input = {"daydate": daydate, "time": time, "venue": venue, "topic": topic, "description": description}
    events = pd.DataFrame.from_dict(pd_input)
    return events


#Initialize user preference file with NULLs
def user_preference(events):
    user_preference = []
    if not os.path.isfile(os.getcwd().replace("\\","/") + "/user_pref.txt"):
        with open("user_pref.txt", "w") as output:
            for row in range(events.shape[0]):
                output.write("NULL" + "\n")
                
    file = open("user_pref.txt", "r") 
    for index, line in enumerate(file): 
            user_preference.append(line.strip("\n")) 
    events["preference"] = user_preference       
    return events


#Add user likes in the user preferences file
def add_likes(events):   
    input_index = int(input("Please enter the index(1-220) of the event that you like. "))
    while input_index-1 not in events.index1:
        print("Invalid option")
        input_index = int(input("Please enter the index(1-220) of the event that you like."))
    events.loc[input_index-1, "preference"] = "like"
    
    with open("user_pref.txt", "w") as output:
            for value in events.preference:
                output.write(value + "\n")
         
    print("Your preferences were added.\n")
    return events


#Recommendation based on tfidf and cosine similarity on event description data
def recommender(events):
    print("Recommended top event is:")
    tfidf = TfidfVectorizer(stop_words = 'english' ).fit_transform(events.description)
    liked = list(events[events.preference == "like"].index)
    nonliked = list(events[events.preference == "NULL"].index)

    #Reference: https://stackoverflow.com/a/12128777
    mean_cosine_similarities = np.mean(linear_kernel(tfidf[nonliked], tfidf[liked]), axis = 1)
    
    top_events = mean_cosine_similarities.argsort()[-1]

    top_index = events.iloc[top_events].index1
    top_topic = events.iloc[top_events].topic
    top_daydate = events.iloc[top_events].daydate
    top_time = events.iloc[top_events].time
    top_venue = events.iloc[top_events].venue

    print("Index:", top_index+1,"Topic:", top_topic,"\nDay & Date:",top_daydate,"Time:",top_time,"Venue:",top_venue)
    return None


#Delete preferences file to delete previous user preferences
def del_pref(events):
    os.remove('user_pref.txt') 
    events['preference']= "NULL"
    print("Removed all saved preferences. You can start afresh now!")
    return events




#Main Function
if __name__ == '__main__':
    events = load_data()
    events = user_preference(events)

    
    events['index1'] = events.index

    #User Menu
    while True:
        print("\n\n\n\n\n********************************************")
        print("Choose one of the following options:")
        print("     Update likes (1)")
        print("     Get reccomendation(Make sure you have liked at least one event) (2)")
        print("     Start afresh (3)")
        print("     Quit (Q)")
        print("**********************************************")

        option = input("Option- ") 

        if option.upper() in ['1','2','3','Q']:
            option = option.upper()
        else:
            print("Invalid option")
            option = None
        
        if option == 'Q':
            print( "Quitting program.")
            break 
        elif option == '1':
            events = add_likes(events)
        elif option == '2':
            recommender(events)
        elif option == '3':
            events = del_pref(events)
        else:
            break