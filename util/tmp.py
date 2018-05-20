l_set=set()
cnt=0
for i in data:
    for j in i['msgCut']:
        l_set.add(str([cnt]+j))
    cnt+=1

l=[eval(i)for i in l_set]

o=[]
while l:
    r=np.random.randint(len(l))
    o.append(l.pop(r))

index=[i[0]for i in o]
lis=[(i[1:]+[0]*100)[:100]for i in o]
lis=lis+lis[:10000]
score=s.get_score(lis)
i2s=list(zip(index,score))
dic={}
for i in i2s:
    if i[0] in dic:
        dic[i[0]].append(i[1])
    else:
        dic[i[0]]=[i[1]]

for i in data:
    i['msgScore']='[]'
for i in dic:
    data[i]['msgScore']=str(dic[i])