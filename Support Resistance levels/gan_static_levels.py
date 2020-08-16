#gann static levels 
import math

gan_static_levels=[]

t=1
start=1
end=1
add=0

for i in range (1,80,1):
    box=i
    if(box==1):
        gan_static_levels.append(box)
        end=1
    else:
        start=end+1
        end=int(math.pow(t+i,2))
        t=t+1
        
        
        for j in range(start+add,end+1,box-1):
            gan_static_levels.append(j)
            
        add=add+1

print(gan_static_levels)

