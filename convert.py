#  This small program extract the useful info from Opensmile output file
#  using for knn graph construction
import random
import math
def getLineNum(file):
    f = open(file,"r")
    line_num = 0
    for line in f.readlines():
        sp = line.split()
        print sp
        line_num += 1
    f.close()
    return line_num
def readFile(src, dest, labelPercentage,featureSize, label):

    w=open(dest,'w+')
    LineNum = getLineNum(src)
    labelNum = int(math.floor(LineNum*labelPercentage))
    rand = set([random.randint(0,LineNum) for i in range(labelNum)])
    index = 0
    f = open(src)
    for line in f:
        newLine = ""
        tmp = line.split(';')
        if index not in rand:
            newLine = "None;"
        else:
            newLine = str(label) + ";"
        for i in range(featureSize):
            newLine = newLine + tmp[i] + ";"
        newLine += "\n"
        w.write(newLine)
        index += 1
    f.close()
    w.close()

readFile("RS_MFCC.txt", "RS_MFCC.csv", 0.1, 30, 1)
readFile("Q_MFCC.txt", "Q_MFCC.csv", 0.1, 30, 2)
readFile("ACK_MFCC.txt", "ACK_MFCC.csv", 0.1, 30, 3)
