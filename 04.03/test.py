s = input("세상에서 가장 쉬운 프로그래밍 언어는? >")
print(s=="파이썬")
x = input("파이썬에서 제곱 연산자는? >")
print(x =="**")
y = input("프로그래밍 언어에서 텍스트를 무엇이라고 부르는가? >")
print(y =="문자열")

import datetime

x = datetime.datetime(2005,4,15)
y = datetime.datetime.now()

delta = y -x 
print("태어난날: "+str(x))
print("현재: "+str(y))
print("태어난지"+str(delta.days)+"일 되었습니다.")

list=[]
list.append(1)
list.append(2)
list.append(6)
list.append(3)
print(list)

subList = ["파이썬","국어","수학","영어"]
print(subList[0])

flist = []
f = input("친구 이름 쓰시오 :")
flist.append(f)
f = input("친구 이름 쓰시오 :")
flist.append(f)
f = input("친구 이름 쓰시오 :")
flist.append(f)
print(flist)

import turtle
t = turtle.Turtle()
t.shape("turtle")
colorList=["red","green"]

t.fillcolor(colorList[0])
t.begin_fill()
t.circle(100)
t.end_fill()
t.fd(100)

t.fillcolor(colorList[1])
t.begin_fill()
t.circle(100)
t.end_fill()
t.fd(100)


import turtle
t = turtle.Turtle()
t.shape("turtle")
CList=[]
C = input("원하는 색을 영어로 입력하시오 : ")
CList.append(C)
t.fillcolor(CList[0])
t.begin_fill()
t.circle(100)
t.end_fill()
t.fd(100)
