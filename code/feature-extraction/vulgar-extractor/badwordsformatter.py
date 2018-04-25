words = []

with open('badwords.txt', 'r') as f:
    for line in f:
        xs = line.split(":")
        x = xs[0]
        words.append(x.replace("\"", ''))

with open('out.out', 'w') as f:      
    for word in words:
        f.write("{}\n".format(word))  