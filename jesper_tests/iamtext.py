from pathlib import Path

fn = Path('./data/iam/iam_lines_gt.txt')

#%%
text = [line for line in fn.read_text().split('\n') if line != '']
names = text[0::2]
lines = text[1::2]
answers = { name: line for name, line in zip(text[0::2], text[1::2]) }

#%%

import re, string
pattern = re.compile('[\W_]+')
#%%

print(pattern.sub('', answers['a03-080-08.png']))
