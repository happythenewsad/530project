import subprocess
import os


def tag(tweets):
    """POS tag list of tweets (as strings), using ARK Twitter Tagger."""

    # get tagger directory path
    path = _get_path()
    input_file = path + "/tweets.txt"

    # write tweets to temporary input file
    _write_temp_input_file(tweets, input_file)

    # run POS tagger and return output
    cmd = ["{}/ark-tweet-nlp-master/runTagger.sh".format(path), input_file]
    FNULL = open(os.devnull, 'w')
    output = str(subprocess.check_output(cmd, stderr=FNULL))
    output = output[2:len(output)-2]

    # delete temporary input file
    os.remove(input_file)

    # read in tagged tweets
    lines = output.split('\\n')
    tagged_lines = []
    for line in lines:
        tokens, tags, *_ = line.split('\\t')
        tokens, tags = tokens.split(), tags.split()
        tagged_line = list(zip(tokens, tags))
        tagged_lines.append(tagged_line)

    return tagged_lines


def tokenize(tweets):
    """Tokenize list of tweets (as strings), using ARK Twokenize."""

    # get tagger directory path
    path = _get_path()
    input_file = path + "/tweets.txt"

    # write tweets to temporary input file
    _write_temp_input_file(tweets, input_file)

    # run POS tagger and return output
    cmd = ["{}/ark-tweet-nlp-master/twokenize.sh".format(path), input_file]
    FNULL = open(os.devnull, 'w')
    output = str(subprocess.check_output(cmd, stderr=FNULL))
    output = output[2:len(output) - 2]

    # delete temporary input file
    os.remove(input_file)

    # read in tagged tweets
    lines = output.split('\\n')
    tokenized_lines = []
    for line in lines:
        tokens, _ = line.split('\\t')
        tokens = tokens.split()
        tokenized_lines.append(tokens)

    return tokenized_lines


def word_count(tagged_line):
    """Return word count of a tagged line."""
    count = 0
    stoptags = ["#", "@", "~", "U", "E", ",", "G"]
    for _, tag in tagged_line:
        if tag not in stoptags:
            count += 1
    return count


def contains_adjectives(tagged_line):
    """Return true if the tagged line contains an adjective."""
    for _, tag in tagged_line:
        if tag == "A":
            return True
    return False


def contains_url(tagged_line):
    """Return true if the tagged line contains a URL."""
    for _, tag in tagged_line:
        if tag == "U":
            return True
    return False


def contains_emoji(tagged_line):
    """Return true if the tagged line contains an emoji."""
    for _, tag in tagged_line:
        if tag == "E":
            return True
    return False


def contains_abbreviation(tagged_line):
    """Return true if the tagged line contains an abbreviation."""
    for _, tag in tagged_line:
        if tag == "G":
            return True
    return False


def _get_path():
    """Get the path of the TwitterParser.py parent folder."""
    cwd = os.getcwd()
    path_extension = "/530project/feature-extraction/twitter-features"
    path = cwd[:cwd.find("530project")] + path_extension
    return path


def _write_temp_input_file(tweets, filename):
    """Write tweets temporary input file for the ARK parser."""
    with open(filename, 'w') as f:
        for tweet in tweets:
            f.write(tweet + '\n')
