f=open('svm_rbf.txt')
i=0
c=[]
tra=[]
tea=[]
for line in f:
    if (i==0):
        i+=1
        s=line.strip().split(',')
        for word in s:
            if '(' in word:
                c.append(round(float(word[2:]),4))
            else:
                tea.append(round(float(word[1:-2]),4))
    else:
        s=line.strip().split(',')
        for word in s:
            if '(' not in word:
                tra.append(round(float(word[1:-2]),4))
import csv
csvf=open('svm_rbf.csv', mode='w')
writer = csv.writer(csvf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
for i in range(len(c)):
    writer.writerow([str(c[i]),str(tra[i]),str(tea[i])])
i=tea.index(max(tea))
writer.writerow([str(c[i]),str(tra[i]),str(tea[i])])