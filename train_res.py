f=open('train_res.txt')
ans=[]
c=1
i=0
cnt=0
for line in f:
    if (c<=48):
        s=line.strip().split()
        for word in s:
            if '0' in word:
                ans.append(0)
            elif '1' in word:
                ans.append(1)
            else:
                ans.append(2)
    else:
        s=line.strip().split()
        if '1' in s[0] and ans[i]==0 or '1' in s[1] and ans[i]==1 or '1' in s[2] and ans[i]==2:
            cnt+=1
    c+=1
print(cnt)