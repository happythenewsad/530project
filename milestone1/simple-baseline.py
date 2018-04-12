import json

# equivalent code can be found in notebooks/exploration.ipynb

#submission = json.load(open('baseline.json', 'r'))
referenceData = json.load(open('../data/dev_data/subtaskB-dev.json', 'r'))


jsn = {}
majorityClass = "false"
for key in referenceData:
    jsn[key] = (majorityClass,1)

with open('baseline.json', 'w') as outfile:
    json.dump(jsn, outfile)