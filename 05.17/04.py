import turtle
t = turtle.Turtle()
t.shape("turtle")

def square(x, y, length1,length2, color):
    t.up()               
    t.goto(x, y)         
    t.down()             
    t.fillcolor(color)    
    t.begin_fill()       
    for i in range(2):
        t.fd(length1)     
        t.left(90)
        t.fd(length2)
        t.left(90)       
    t.end_fill()        

square(-200, 0, 100,200,"red")
square(0, 0, 100, 200,"blue")
square(200, 0, 100, 200,"yellow")

turtle.done()



