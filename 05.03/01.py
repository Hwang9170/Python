import time
import random
import turtle

t = turtle.Turtle()
t.shape = ("turtle")

for i in range(10,0,-1):
  print(i)
  time.sleep(1)
print("발사")

dan = int(input("원하는 단은:"))
for i in range(1,10,1):
  print(f"{dan}*{i}={dan*i}")

print("구구단입니다.")
for dan in range(1,10,1):
 for i in range(1,10,1):
  print(f"{dan}*{i}={dan*i}")

import time 
import winsound

sec = int(input("초 단위의 시간을 입력하시오 : "))

for i in range(sec,0,-1):
  print(f"{i}초 남았습니다.")
  time.sleep(1)
winsound.Beep(2000,300)

for i in range(30):
 length = random.randint(1,100)
 t.fd(length)
 angle = random.randint(-180,180)
 t.right(angle)
turtle.done

i = 0 
while i <5:
 t.fd(50)
 t.right(144)
 i +=1
turtle.done 

f = 1
n = int(input("숫자입력:"))
for i in range(1,n+1):
 f = f*i
print(n,"팩토리얼은",f)

re = "아니"
while re == "아니":
  print("아니")
  re = input("다됐어?")
  while re == "먹자":
    print("됐어 !")

total = 0
answer = "yes"

while answer == "yes" :
    num = int(input("숫자입력 : "))
    total += num
    print(f"현재 숫자 {total}")
    answer = input("계속?(yes or no) : ")
else :
    print("합계 : " , total)