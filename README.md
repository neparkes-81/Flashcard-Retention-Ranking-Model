# Flashcard Retention Ranking Model

In language learning, flashcards can be an effective tool to reinforce new vocabulary or retain old knowledge, but various factors impact a learners ability to recall terms. **This project seeks to** create a ranking system to predict the binary classification of learner likelihood to recall words as they appear in flashcards. The ranking model will take in both time-related and linguistic features to determine said likelihood and decide which word to put next in a series of flashcards considering a scale of high-priority (hard or unlikely to recall) to low-priority (easy or very likely to recall) words. 

Smart flashcard tools exist such as Anki which specifically relies on time-related features and user given feedback of difficulty. **Yet, I propose that the consideration of linguistic features related to a word can improve the algorithm's performance identifying optimal word rank in a flashcard series.** I take inspiration from ad ranking systems. Here, a word is akin to a user where not only activity, but also demographic data is analyzed.

## Dataset
I am using the **Duolingo 2018 Second Language Acquisition Modeling (SLAM) dataset**. This is a corpus of app user results for back-translation exercises (providing features such as time taken to complete, word level linguistic, characteristics, etc.). I specifically focus on the subset for English-speaking learners of French `fr_en` consisting of over 900,000 productions and thus, I see to predict learner ability to recall French words.

## Project Set Up

The project is organized into three part: **exploratory data analysis**, **feature engineering**, and **ranking model training/ testing**. The files are label respectively:

 - `eda.ipynb` - *completed*
 - `feature_engineering.ipynb` - *in devlopment*
 - `model.ipynb` - *coming soon*

## Citations
    B. Settles, C. Brust, E. Gustafson, M. Hagiwara, and N. Madnani. 2018.
    Second Language Acquisition Modeling.
    In Proceedings of the NAACL-HLT Workshop on Innovative Use of NLP for Building Educational Applications (BEA). ACL.