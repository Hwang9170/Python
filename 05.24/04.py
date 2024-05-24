import turtle

t = turtle.Turtle()
t. shape("turtle")


def draw():
   t.fd(100)
   t.backward(100)

for i in range(12):
  draw()
  t.right(30)

  #2