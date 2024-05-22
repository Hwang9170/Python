import turtle
t = turtle.Turtle()
t.shape("turtle")

def square(x, y, length, color):
    t.up()               
    t.goto(x, y)         
    t.down()             
    t.fillcolor(color)    
    t.begin_fill()       
    for i in range(4):
        t.fd(length)     
        t.left(90)   
    t.end_fill()        

square(-200, 0, 100,"red")
square(0, 0, 100,"blue")
square(200, 0,100,"yellow")

turtle.done()



