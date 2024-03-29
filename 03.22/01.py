number = int(input("정수를 입력하시오:"))
print(number%2)

p = int(input("분자를 입력하시오>"))
q = int(input("분모를 입력하시오>"))

print("나눗셈의 몫 =",p//q)
print("나눗셈의 나머지 = ",p%q)

import turtle
turtle.Turtle()
t = turtle.Turtle()

rangea = int(input("몇각형의 원을 그리시겠습니까? :"))

for i in range(rangea):
  t.fd(100)
  t.left(360//rangea)

turtle.done()

#아메리카노2000 + 카페라테 3000 카푸치노 3500

ame = 2000
cafera = 3000
cafu = 3500

ame1 = int(input("아메리카노 팔린 개수 : "))
cafera2= int(input("카페라테 팔린 개수 : "))
cafu2 = int(input("카푸치노 팔린 개수 : "))

sum = ame*ame1 +cafera*cafera2 + cafu*cafu2
print("총매출은",sum,"원 입니다.")

humm = int(input("재료비 :"))
print("순 수익은",sum - humm,"입니다.")

hwa = int(input("화씨 온도를 입력하시오 :"))
sup = (hwa-32.0)*5.0/9.0
print("화씨 온도:",hwa)
print("섭씨 온도:",sup)

sup = int(input("섭씨 온도를 입력하시오 :"))
hwa = (sup+32.0)*9.0/5.0
print("화씨 온도:",hwa)
print("섭씨 온도:",sup)

won = int(input("돈을 넣으시오:"))
price = int(input("물건 가격: "))

change = won - price
print("거스름돈 :",change)

c500 = change//500
change = change%500
c100 = change//100
change = change%100
c50 = change//50

print("500원 동전 수",c500)
print("1000원 동전 수 ", c100)
