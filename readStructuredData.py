#!python

import sys
import csv
import os.path
import pickle
import datetime
from os.path import join

csvfile = ""
working_dir = ""
f = open(csvfile, 'rU')
reader = csv.reader(f, delimiter=",")
reader.next()#skip the header line
pidDate_ICDs={}
pidDate_CPTs={}
for row in reader:
        pid= row.pop(0)
        date = row.pop(0)
        cpt = row[-1]
        del row[-1] # remove the last element --cpt code
        #massage date to the right form:
        slash = date.rfind("/")
        date1 = date[0:slash]
        year = date[slash+1:]
        if 59 <= int(year) <= 99:
            year = '19' + year
        else:
            year = '20' + year
        date = date1+"/"+year
        d = datetime.datetime.strptime(date, "%m/%d/%Y").date()
        if (pid,d) not in pidDate_ICDs.keys():
            ICDs = []
            CPTs = []
            for code in row:
                if code:
                    ICDs.append(code)
            CPTs.append(cpt)
            pidDate_ICDs[(pid,d)]=ICDs
            pidDate_CPTs[(pid,d)]=CPTs
        else:
            ICDs = pidDate_ICDs.get((pid,d))
            for code in row:
                if code:
                    ICDs.append(code)
            pidDate_ICDs[(pid,d)] = ICDs
            CPTs = pidDate_CPTs.get((pid,d))
            CPTs.append(cpt)
            pidDate_CPTs[(pid,d)]=CPTs

pickle.dump(pidDate_ICDs, open(os.path.join(working_dir, 'pidDate_ICDs_dic.p'),"wb"))
pickle.dump(pidDate_CPTs, open(os.path.join(working_dir, 'pidDate_CPTs_dic.p'),"wb"))

with open(os.path.join(working_dir, 'pidDate_ICDs_dic.csv'),"wb") as icdfile:
    icdwriter = csv.writer(icdfile, delimiter=",")
    for (pid,d) in pidDate_ICDs.keys():
        icdwriter.writerow([pid, d] + pidDate_ICDs.get((pid,d)))
with open(os.path.join(working_dir, 'pidDate_CPTs_dic.csv'),"wb") as cptfile:
    cptwriter = csv.writer(cptfile, delimiter=",")
    for (pid,d) in pidDate_CPTs.keys():
        cptwriter.writerow([pid, d] + pidDate_CPTs.get((pid,d)))

#find all raw text files:
rawTextDir = ""
doc2ICDs = {}
doc2CPTs = {}
pids = os.listdir(rawTextDir)
fileNum = 0
for pid in pids:
    for file in os.listdir(join(rawTextDir,pid)):
        fileNum=fileNum+1
        #massage the file name to get the date:
        runder = file.rfind("_")
        date = file[0:runder]
        runder = date.rfind("_")
        date = date[runder+1:]
        d = datetime.datetime.strptime(date, "%m-%d-%Y").date()
        if (pid,d) not in pidDate_ICDs.keys():
            print "no codes for %s on %s for file %s" % (pid, d, file)
            # ICDs = []
            # CPTs = []
            # doc2ICDs[file]=ICDs
            # doc2CPTs[file]=CPTs
        else:
            ICDs = pidDate_ICDs.get((pid,d))
            CPTs = pidDate_CPTs.get((pid,d))
            doc2ICDs[file]=ICDs
            doc2CPTs[file]=CPTs
pickle.dump(doc2ICDs, open(os.path.join(working_dir, 'doc2ICDs_dic.p'),"wb"))
pickle.dump(doc2CPTs, open(os.path.join(working_dir, 'doc2CPTs_dic.p'),"wb"))

with open(os.path.join(working_dir, 'docName_ICDs_dic.csv'),"wb") as icdfile:
    icdwriter = csv.writer(icdfile, delimiter=",")
    for name in doc2ICDs.keys():
        icdwriter.writerow([name] + doc2ICDs.get((name)))

cuiFileDir = ""
nonEmptyFiles = os.popen('find '+cuiFileDir+' -type f ! -size 0').readlines()
file_cuis = {}
for file in nonEmptyFiles:
    file = file.rstrip()
    cuis = set(line.strip() for line in open(file))
    file_cuis[file] = cuis
with open(os.path.join(working_dir, 'docName_CUIs_dic.csv'),"wb") as cuifile:
    cuiwriter = csv.writer(cuifile, delimiter=",")
    for name in file_cuis.keys():
        idx=name.rfind("/")
        cuiwriter.writerow([name[idx+1:]] + list(file_cuis.get(name)))
