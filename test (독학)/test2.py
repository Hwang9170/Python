a = int(input("숫자 입력하세요 > "))
print(a)
r = float(input("반지름을 입력하시오 >"))
pi = 3.141592
ra = r**2*pi
print("원의 넓이 =", ra)

import turtle

t = turtle.Turtle()
t.shape("turtle")

radius = int(input("반지름을 입력하시오 >"))
color = input("색을 입력하시오 >")

t.color(color)
t.begin_fill()
t.circle(radius)
t.end_fill()

t = int(input("시간 입력>"))
m = 340*t
print(m)

import turtle
t = turtle.Turtle()
t.shape("turtle")
t.goto(0,0)
t.setheading(45)
t.fd(141.4)
t.up()
t.goto(0,0)
t.down()
t.setheading(0)
t.forward(100)
t.setheading(90)
t.forward(100)

import time
f =time.time()
print(f)
f = int(f)
h = f//60//60%24+9
m = f//60%60

print("지금 시간은"+str(h)+"시"+str(m)+"분 입니다.")

price = int(input("물건 가격을 입력해주세요 >"))
money = int(input("돈을 넣어주세요 >"))
qm = money-price
print("거스름돈은",qm)
#동전계산
c500 = qm//500
qm = qm%500
c100 = qm//100

print("500원",c500,"100원",c100)

x = 3
y = 3.14
z = "3"
print(type(x))
print(type(y))
print(type(z))

print("안녕"+str(x))

k = "안녕반갑다"
print(k[0:2])

