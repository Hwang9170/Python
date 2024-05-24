import turtle
import random

t = turtle.Turtle()
t.shape("turtle")

def sq(length):
  for i in range(4):
    t.fd(length)
    t.left(90)

def draw(x,y):
  t.penup()
  t.goto(x,y)
  t.pendown()
  t.begin_fill()
  t.color(random.random(),random.random(),random.random())
  sq(100)
  t.end_fill()
s = turtle.Screen()
s.onscreenclick(draw)

turtle.done()