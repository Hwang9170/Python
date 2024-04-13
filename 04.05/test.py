score = 70
if score >= 60:
  print("합격입니다.")

else:
  print("불합격 입니다.")

import turtle
t=turtle.Turtle()
t.shape("turtle")

t.penup()
t.goto(100,100)
t.write("거북이가 여기로 오면 양수 입니다.")
t.goto(100,0)
t.write("거북이가 여기로 오면 0입니다.")
t.goto(100,-100)
t.write("거북이가 여기로 오면 음수입니다.")

t.goto(0,0)
t.pendown()
s = turtle.textinput("","숫자를 입력하시오:")

n = int(s)
if(n>0):
  t.goto(100,100)
if(n==0):
  t.goto(100,0)
if(n<0):
  t.goto(100,-100)

  #영화의 가격은 ~ 원 입니다. / 다른 영화를 보시겠습니까?

age = int(input("나이를 입력하시오 :"))

if age >=19:
  print("이 영화를 보실 수 있습니다. \n 영화의 가격은 10000원 입니다. ")
  
else: 
  print("이 영화를 보실 수 없습니다.\n 다른 영화를 보시겠어요?")



age = int(input("나이를 입력하시오 : "))
cm = int(input("키를 입력하시오(cm) : "))

if age>10 and cm >165:
  print("놀이기구를 탈 수 있습니다.")

else:
  print("놀이기구를 타실 수 없습니다. ")

  