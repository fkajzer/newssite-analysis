import json
from collections import Counter
from pprint import pprint

#with open('data/faz/faz.json') as data_file:
#    data = json.load(data_file)

with open('data/zeit/zeit.json') as data_file:
    data = json.load(data_file)

#with open('data/spiegel/spiegel.json') as data_file:
#    data = json.load(data_file)
#pprint(data[0])

c = Counter(comment['user_name'] for comment in data)

f = open('resul_zeit.py', 'w' )
f.write(repr(c))
f.close()
