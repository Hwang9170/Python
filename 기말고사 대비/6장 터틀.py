import turtle 
import random
t = turtle.Turtle()
t.shape("turtle")
for i in range(6):
  t.circle(100)
  t.left(60)

i = turtle.textinput("","몇각형?")
k = int(i)
for i in range(k):
  t.forward(100)
  t.left(360/k)

for i in range(30):
  number = random.randint(1,100)
  turtle.fd(number)
  left = random.randint(-180,180)
  turtle.left(left)

for i in range(6):
  t.fd(200)
  t.right(144)

i = 0 
while i <5:
  t.fd(200)
  t.right(144)
  i +=1

turtle.done()


