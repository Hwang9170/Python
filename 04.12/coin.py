import turtle
import random

screen = turtle.Screen()

image1 = "C:\\front.gif"
image2 = "C:\\back.gif"
screen.addshape(image1)
screen.addshape(image2)

t1 = turtle.Turtle()
coin = random.randint(0,1)
if coin == 0:
  t1.shape(image1)
  t1.circle(100)
  t1.stamp()
else: 
  t1.shape(image2)
  
t1.circle(100)
t1.stamp
