
hi = "hello world!"
print(hi.upper()) #대문자로 변경
print(hi.lower()) #소문자로 변경
print(hi.swapcase()) #대문자는 소문자로 소문자는 대문자로 
print(hi.title()) #앞 글자들만 대문자로 

python = "   파   이   썬   "
python.strip()
python.rstrip() 
python.lstrip()
python.replace("파","카프리")

print(python.strip()) #양쪽 공백 제거 
print(python.rstrip()) #오른쪽 공백 제거
print( python.lstrip()) #왼쪽 공백 제거 
print( python.replace("파","카프리"))#파-> 카프리로 변경 

total = 0
for num in [10,20,30,40,50]:
  total =total+num
print("합계:",total)
print("----------")
total = 0
for num in [10,20,30,40,50]:
  total =total+num
  print("합계:",total)

total = 0
numbers = [10,20,30,40,50]
for num in numbers:
  total =total+num
print("합계:",total)

solar = ["수성","금성","지구","화성"]
count = 1
for star in solar:
  print("%s은 %d번째 행성"%(star,count))
  count = count+1

for count in range(5,10):
  print("안녕 count = ",count)
print("----------")
for count in range(3,15,3):
  print("안녕 count = ",count)
print("-----")

for i in range(5,10):
  print(f"변수의 i의 값={i}")
print()
for i in range(1,10,3):
  print(f"변수의 i의 값={i}")
print()

for i in range(10,7,-1):
  print(f"변수의 i의 값={i}")
print()

import turtle
t = turtle.Turtle()
t.shape("turtle")

radius =100
t.circle(radius)
t.fd(50)
radius = radius+50
t.circle(radius)
t.fd(50)
radius = radius+50
t.circle(radius)
t.fd(50)
radius = radius+50
t.circle(radius)

foodlist = []
food = input("음식이름>>")
foodlist.append(food)
food = input("음식이름>>")
foodlist.append(food)
food = input("음식이름>>")
foodlist.append(food)
food = input("음식이름>>")
foodlist.append(food)
print(foodlist)

for i in range(6):
 t.circle(100)
 t.left(60)

import turtle
t = turtle.Turtle()
t.shape("turtle")

clist = ["red","green","blue","yellow"]


for i in range(4):
  t.fillcolor(clist[i])
  t.begin_fill()
  t.circle(100)
  t.end_fill()
  t.fd(100)

number = turtle.textinput("","몇각형을 원하시나요?")
number = int(number)

for i in range(number):
  t.fd(100)
  t.left(360/number)
t.done()

numstring = input("여러개 정수 입력 > ")
print("-"*40)
numstringlist = numstring.split()
print(numstringlist)
print("-"*40)

import time

for i in range(10,0,-1):
  print(i)
  time.sleep(1)
print("발사!")

number = int(input("원하는 단은 >>"))
for i in range(1,10):
  print(f"{number}*{i}=",number*i)

  count = 90
while count<=100:
  print(f"hi{count:05d}")
  count = count +1
print("종료")

total = 0 
count = 1
while count<11:
  total= total+count
  count=count+1
print(total)

password = ""
while password != "1234":
 password = input("암호를 입력하시오 >>")
print("로그인 성공 ")

import random

t = 0
n = 0 
Q = random.randint(1,100)
print("1부터 100 사이의 숫자를 맞추시오")

while n != Q:
 n = int(input("숫자를 입력하시오>>"))
 t  = t+1
 if n < Q:
  print("더 크게 !")
if n > Q:
 print("더 작게 !")

if n == Q:
 print("정답입니다. 시도 횟수는 = ",t)

 radius = 100

a = 1
while a <=3:
  t.circle(radius)
  t.fd(50)
  radius = radius+50
  a =a+1

  radius = 100
k = 1
while k<=6:
  t.circle(radius)
  t.left(60)
  k = k+1

  name = input("이름 입력>>")
print(f"안녕하세요{name}님")