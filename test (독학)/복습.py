print("100"+"200")
print(100+200)

print("반가워요"*30)

import turtle
t = turtle.Turtle()
t.shape("turtle")

t.forward(100)
t.right(90)
t.forward(100)
t.right(90)
t.forward(100)
t.right(90)
t.forward(100)
t.right(90)

for i in range(6):
  t.fd(100)
  t.right(60)

import turtle
colors = ["red","purple","green","yellow","orange"]
t = turtle.Turtle()

turtle.bgcolor("black")
t.speed(0)
t.width(3)
length = 10

while length<500:
  t.forward(length)
  t.pencolor(colors[length % len(colors)])
  t.right(89)
  length+=5

print("안녕하세요? 여러분")
print("저는 파이썬을 무척 좋아합니다.")
print("9*8은",9*8,"입니다.")

t.forward(100)
t.right(120)
t.forward(100)
t.right(120)
t.forward(100)
t.right(120)

x1 = int(input("정수를 입력하시오:"))
x2 = int(input("정수를 입력하시오:"))
print(x1,"과",x2,"의 합은",x1+x2)

radius = 100
t.fd(100)
t.circle(radius)
t.fd(100)
t.circle(radius)
t.fd(100)
t.circle(radius)

name = input("이름을 입력하시오 :")
print(name,"씨 안녕하세요?")
print("파이썬에 세계에 오신걸 환영합니다.")
x1 = int(input("첫 번째 정수를 입력하시오 : "))
y1 = int(input("두 번째 정수를 입력하시오 : "))
print(x1,"과",y1,"의 합은",x1+y1,"입니다.")

size = int(input("집의 크기를 얼마로 할까요?"))

t.fd(size) 
t.right(90)
t.fd(size) 
t.right(90)
t.fd(size) 
t.right(90)
t.fd(size) 
t.right(90)
t.fd(size) 
t.left(120)
t.fd(size) 
t.left(120)
t.fd(size) 
t.left(120)

where = input("경기장은 어디입니까?>")
win = input("이긴 팀은 어디입니까?>")
lose = input("진 팀은 어디입니까?>")
mvp = input("우수 선수는 누구입니까?>")
score = input("스코어는 몇대몇 입니까?>")

print("오늘",where,"에서 야구 경기가 열렸습니다.")
print(win,"(과/와)",lose,"가 치열한 공방전을 펼쳤습니다.")
print(mvp,"(이/가) 맹활약 하였습니다.")
print("결국",win,"(이/가)",lose,"를",score,"으로 이겼습니다.")

name = turtle.textinput("","이름을 입력하세요")
t.write("안녕하세요"+name+"님")

for i in range(4):
  t.fd(100)
  t.right(90)

print("안녕하세요?")
name = input("이름이 어떻게 되시나요?>")
print("만나서 반갑습니다.{}씨".format(name))
print("이름의 길이는 다음과 같군요",len(name))
age = int(input("나이가 어떻게 되나요?>"))
print("내년이면",str(age+1),"되시는 군요")
hobby = input("취미가 무엇인가요?")
print("네 저도",hobby,"좋아합니다.")

y =int(input("오늘의 연도를 입력하시오 : "))
m =int(input("오늘의 월을 입력하시오 : "))
d = int(input("오늘의 일을 입력하시오 : "))
print("오늘은",y,"년",m,"월",d,"일 입니다.")

import time
now = time.time()
thisYear = int(1970+now//(365*24*3600))
thisYear = str(thisYear)
print("올해는",thisYear,"입니다.")
age = int(input("올해 몇 살이신지요?"))
thisYear = int(thisYear)
print("2050년에는"+str(age+2050-thisYear)+"살 이시군요")

flist = []
f = input("친구 이름을 입력하시오")
flist.append(f)
f = input("친구 이름을 입력하시오")
flist.append(f)
f = input("친구 이름을 입력하시오")
flist.append(f)
f = input("친구 이름을 입력하시오")
flist.append(f)
f = input("친구 이름을 입력하시오")
flist.append(f)
print(flist)

list = ["red","green","black"]


t.fillcolor(list[0])
t.begin_fill()
t.circle(100)
t.end_fill()
t.fd(100)
t.fillcolor(list[1])
t.begin_fill()
t.circle(100)
t.end_fill()
t.fd(100)
t.fillcolor(list[2])
t.begin_fill()
t.circle(100)
t.end_fill()
t.fd(100)

