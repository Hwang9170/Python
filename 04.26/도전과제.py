import turtle
t = turtle.Turtle()
t.shape("turtle")

s = turtle.textinput("알림","몇각형을 원하시나요?")
n= int(s)

for i in range(n):
  t.fd(100)
  t.left(360/n)
t.done()
