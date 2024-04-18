import turtle
t = turtle.Turtle()
t.shape("turtle")
t.width(10) ##두께 
t.color("red") ##색상

for a in range(4):
 t.forward(100)
 t.right(90)
t.penup()

t.goto(100,100)
t.pendown()
for r in range(3):
 t.forward(100)
 t.left(120)
t.penup()
t.goto(200,200)
t.pendown()
t.circle

