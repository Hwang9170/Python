
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