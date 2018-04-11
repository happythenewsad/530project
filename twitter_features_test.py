import os
import sys

sys.path.append(os.getcwd() + "/feature-extraction/twitter-features")
from TwitterParser import TwitterParser    # noqa


example_tweets = [
    "I predict I won't win a single game I bet on. Got Cliff Lee today, so if he loses its on me RT @e_one: Texas (cont) http://tl.gd/6meogh",
    "RT @DjBlack_Pearl: wat muhfuckaz wearin 4 the lingerie party?????",
    "Wednesday 27th october 2010. 》have a nice day :)",
    "RT @ddlovato: @joejonas oh, hey THANKS jerk!",
    "@thecamion I like monkeys, but I still hate COSTCO parking lots..",
    "@DDaimaru I may have to get minecraft after watching videos of it",
    "RT @eye_ee_duh_Esq: LMBO! This man filed an EMERGENCY Motion for Continuance on account of the Rangers game tonight! « Wow lmao",
    "RT @musicdenver: Lady Gaga - Bad Romance http://dld.bz/n6Xv",
    "RT @cheriexamor: When you have a good thing, hold it, squeeze it, never let it go.",
    "Texas Rangers are in the World Series!  Go Rangers!!!!!!!!! http://fb.me/D2LsXBJx",
    "I hope this goes well. Beep boop. @Robots97",
    "Yayyy! 100 tix sold :D ily all!",
    "@RealDonaldTrump you suckkkk mothafucka #impeach",
    "39% of my nominations, including Diplomats to foreign lands, have not been confirmed due to Democrat obstruction and delay. At this rate, it would take more than 7 years before I am allowed to have these great people start working. Never happened before. Disgraceful!",
    "Oklahoma Leaders Claim Teachers' Strike Betrays Values Of Nation's 1914 Founding By Abraham Lincoln And Orville Redenbacher https://trib.al/ullIvAS",
    "One of the 2 new sticker sheets I made! Hoping @ZapCreatives will get to my order soon! ^^ I'm hoping to get this by April 23rd! :):):)",
    "Today, it was my honor to welcome Estonia President @KerstiKaljulaid, Lithuania President @Grybauskaite_LT, and Latvia President @Vejonis to the @WhiteHouse. Congratulations on your 100th anniversaries of independence! #BalticSummit: http://45.wh.gov/RtVRmD "
]


tagged_tweets = TwitterParser.tag(example_tweets)
for tweet in tagged_tweets:
    print(tweet)
    print("N:\t\t{}\nurl:\t{}\nadj:\t{}\nemoji:\t{}\nabbr:\t{}\n".format(
        TwitterParser.word_count(tweet),
        TwitterParser.contains_url(tweet),
        TwitterParser.contains_adjectives(tweet),
        TwitterParser.contains_emoji(tweet),
        TwitterParser.contains_abbreviation(tweet)),
        TwitterParser.pos_counts(tweet)
    )
