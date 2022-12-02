import sys

#main
str = "wrd"

state = 0

for i in range(state+1,len(str)+1):
    out = str[state:i]
    print(out)