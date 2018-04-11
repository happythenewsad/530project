import subprocess
import os


class TwitterParser:
    tagset = ['N', 'O', 'S', '^', 'Z', 'L', 'M', 'V', 'A', 'R', '!', 'D', 'P',
              '&', 'T', 'X', 'Y', '#', '@', '~', 'U', 'E', '$', ',', 'G']

    @staticmethod
    def tag(tweets):
        """POS tag list of tweets (as strings), using ARK Twitter Tagger."""

        # get bash script and temporary input file directory paths
        script_file = os.path.dirname(__file__) + "/ark-tweet-nlp/runTagger.sh"
        input_file = os.path.dirname(__file__) + "/tweets.txt"

        # write tweets to temporary input file
        with open(input_file, 'w') as f:
            for tweet in tweets:
                f.write(tweet + '\n')

        # run ARK tagger and return output
        cmd = [script_file, input_file]
        DEVNULL = open(os.devnull, 'w')
        output = str(subprocess.check_output(cmd, stderr=DEVNULL))
        output = output[2:len(output)-2]

        # read in tagged tweets
        lines = output.split('\\n')
        tagged_lines = []
        for line in lines:
            tokens, tags, *_ = line.split('\\t')
            tokens, tags = tokens.split(), tags.split()
            tagged_line = list(zip(tokens, tags))
            tagged_lines.append(tagged_line)

        # delete temporary input file
        os.remove(input_file)

        return tagged_lines

    @staticmethod
    def tokenize(tweets):
        """Tokenize list of tweets (as strings), using ARK Twokenize."""

        # get bash script and temporary input file directory paths
        script_file = os.path.dirname(__file__) + "/ark-tweet-nlp/twokenize.sh"
        input_file = os.path.dirname(__file__) + "/tweets.txt"

        # write tweets to temporary input file
        with open(input_file, 'w') as f:
            for tweet in tweets:
                f.write(tweet + '\n')

        # run ARK tokenizer and return output
        cmd = [script_file, input_file]
        DEVNULL = open(os.devnull, 'w')
        output = str(subprocess.check_output(cmd, stderr=DEVNULL))
        output = output[2:len(output) - 2]

        # read in tagged tweets
        lines = output.split('\\n')
        tokenized_lines = []
        for line in lines:
            tokens, _ = line.split('\\t')
            tokens = tokens.split()
            tokenized_lines.append(tokens)

        # delete temporary input file
        os.remove(input_file)

        return tokenized_lines

    @staticmethod
    def pos_counts(tagged_line):
        counts = [0] * len(TwitterParser.tagset)
        for _, tag in tagged_line:
            counts[TwitterParser.tagset.index(tag)] += 1
        return counts

    @staticmethod
    def word_count(tagged_line):
        """Return word count of a tagged line."""
        count = 0
        stop_tags = ['#', '@', '~', 'U', 'E', ',', 'G']
        for _, tag in tagged_line:
            if tag not in stop_tags:
                count += 1
        return count

    @staticmethod
    def contains_adjectives(tagged_line):
        """Return true if the tagged line contains an adjective."""
        for _, tag in tagged_line:
            if tag == "A":
                return True
        return False

    @staticmethod
    def contains_url(tagged_line):
        """Return true if the tagged line contains a URL."""
        for _, tag in tagged_line:
            if tag == "U":
                return True
        return False

    @staticmethod
    def contains_emoji(tagged_line):
        """Return true if the tagged line contains an emoji."""
        for _, tag in tagged_line:
            if tag == "E":
                return True
        return False

    @staticmethod
    def contains_abbreviation(tagged_line):
        """Return true if the tagged line contains an abbreviation."""
        for _, tag in tagged_line:
            if tag == "G":
                return True
        return False