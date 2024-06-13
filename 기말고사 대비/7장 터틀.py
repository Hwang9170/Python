import turtle
t = turtle.Turtle()
t.shape("turtle")

def s(x,y,l,c):
  t.up()
  t.goto(x,y)
  t.down()
  t.fillcolor(c)
  t.begin_fill()
  for i in range(4):
    t.forward(l)
    t.left(90)
  t.end_fill()
s(0,0,100,"yellow")
s(200,0,100,"red")

turtle.done()