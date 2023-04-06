import os

file = open("shell.txt")
while 1:
    line = file.readline()
    if not line:
        break
    os.system(line)
file.close()