import subprocess
import os


def twitter_pos_tag(tweets):
    """POS tag a list of tweets as strings, using ARK Twitter Tagger."""

    # get tagger directory path
    cwd = os.getcwd()
    path = cwd[:cwd.find("530project")] + "/530project/feature-extraction/POS"

    # write tweets to temporary input file
    with open('{}/tweets.txt'.format(path), 'w') as f:
        for tweet in tweets:
            f.write(tweet + '\n')

    # run POS tagger and return output
    cmd = [
        "{}/ark-tweet-nlp-master/runTagger.sh".format(path),
        "{}/tweets.txt".format(path)
    ]
    output = str(subprocess.check_output(cmd))
    output = output[2:len(output)-2]

    # delete temporary input file
    os.remove("{}/tweets.txt".format(path))

    # read in tagged tweets
    lines = output.split('\\n')
    tagged_lines = []
    for line in lines:
        tokens, tags, confidences, tweet = line.split('\\t')
        tokens, tags = tokens.split(), tags.split()
        tagged_line = list(zip(tokens, tags))
        tagged_lines.append(tagged_line)

    return tagged_lines


if __name__ == '__main__':
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
    ]

    example_tags = twitter_pos_tag(example_tweets)
    print(example_tags)
