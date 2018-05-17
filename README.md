# Event-Recommender-System for UIUC

Language used: python3
Packages required: numpy, pandas, sklearn, os

## Functionality

A list of events with its attributes like date, time, venue etc is present in events.csv. This events.csv is split into various text files with one column in each. On running the final.py file the user gets the following menu:
```
***************************************************************************
Choose one of the following options:
     Update likes (1)
     Get reccomendation(Make sure you have liked at least one event) (2)
     Start afresh (3)
     Quit (Q)
****************************************************************************
```
Option 1 : On selecting option the user will be asked to update the event index that he likes. User can choose from events.csv or topic.txt for heading of the events. This is step has to be done atleast once in order to get recommendation.

Option 2 : Using tf-idf vectorizer on description of the events, cosine similarity is calculated and the most relevant event is recommended. 

Option 3 : If the user wishes to remove his or her previous likes history and wants to start afresh.

## Implementation

The implementation is done in python using Pandas data frame and using the Description column to get cosine similarity between liked events and non liked event and scikit-learn is used for the same.

## Usage

```
python3 final.py
```
