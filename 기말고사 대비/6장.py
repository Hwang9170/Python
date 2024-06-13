for i in range(3):
  print("안녕")
for i in [1,2,3,4,5]:
  print("i =",i)
  print("hi")
for i in [5,4,3,2,1]:
  print("Hi")
  print("i =",i)

sum = 0 
for i in [1,2,3,4,5]:
  sum +=i 
print(sum)

for i in range(1,6,1):
  print(i,end="") #end = "" 줄바꿈 없이 출력 

import time

for i in range(10,0,-1):
  print(i)
  time.sleep(1)
print("발사 !!")

t = int(input("초 단위의 시간을 입력하시오:"))
for i in range(t,0,-1):
  print(f"{i}초 남았습니다.")
  time.sleep(1)
print("뭐 여기 알람 넣으면 되겠지..")

dan = int(input("원하는 단은?"))
for i in range(1,10,1):
  print(f"{dan}*{i}={dan*i}")
  