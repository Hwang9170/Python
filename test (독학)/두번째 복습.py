name = input("이름을 입력하시오>>")
print(f"안녕하세요{name}님!")

x = 100
x = 200
print(x)

x = 7
y = 3
print(x+y)
x = "7"
y = "3"
print(x+y)

name = input("이름 입력 >>")
age = int(input("나이 입력>>"))
age = age +1
print(f"{name}씨는 내년에{age}살 이시네요")

w = input("주소>>")
r = int(input("방 개수>>"))
p = int(input("가격>>"))
print(f"{w}에 위치한 {r}개의 방을 가진 아파트 단돈 {p}원")

one = int(input("정수 입력 >>"))
two = int(input("정수 입력 >>"))
three = int(input("정수 입력 >>"))
print("입력하신 정수의 평균값은",(one+two+three)/3,"입니다.")

r = int(input("반지름 입력 >>"))
print(f"반지름이 {r}인 원의 넓이 =",r**2*3.14)

import turtle
r= 100

turtle.up()
turtle.goto(0,0)
turtle.down()
turtle.circle(r)

r = r+10 
turtle.up()
turtle.goto(100,0)
turtle.down()
turtle.circle(r)

r = r+10
turtle.up()
turtle.goto(200,0)
turtle.down()
turtle.circle(r)

print("----------")

import turtle
t = turtle.Turtle()
t.shape("turtle")

range1 = int(input("몇각형?>>"))
for i in range(range1):
  t.forward(100)
  t.right(360//range1)
turtle.done()

ame = int(input("아메리카노 팔린 개수 :"))
c = int(input("카페라테 팔린 개수 :"))
cf = int(input("카푸치노 팔린 개수 :"))

num = ame*2000 + c*3000 +cf*3500 
print(f"총 매출은 {num}원 입니다.")

i = int(input("재료비>>"))
print("순수익 = " ,num-i)

h = float(input("화씨 >>"))
s =(h-32.0)*5.0/9.0
print(f"화씨 = {h}")
print(f"섭씨 = {s}")

m = int(input("돈을 넣으세요 >>"))
p = int(input("물건 가격 >>"))

c = m-p 
print("거스름돈 = ",c)
c500 =c//500
c100= (c%500)//100
c50= (c%100)//50
print(f"500원 = {c500}| 100원 = {c100} 50원 = {c50}")

x = int(input("정수>>"))
y = int(input("정수>>"))
z = int(input("정수>>"))
print("평균",(x+y+z)/3)

print("문제")
x = int(input("1+1=? 답>>"))
print(x==1+1)

import math 
radius = 3
circle = 2*math.pi*radius
print("7! = ",math.factorial(7))
print("6.99999와 7 비교 ", math.isclose(6.99999,7))
print("log(3.4)=",math.log(3,4))
print("4제곱근",math.sqrt(4.0))

A = "가나다"
B = "마바사"
print(A<B)
c = "cristiano"
R = "ronaldo"
print(c<R) ##알파벳 순인듯




