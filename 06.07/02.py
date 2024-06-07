from tkinter import*

window=Tk()

w = Button(window,text="박스 #1",bg="red",fg="white")
w.place(x=0,y=0)
w = Button(window,text="박스 #2",bg="gray",fg="white")
w.place(x=20, y=20)
w = Button(window,text="박스 #3",bg="black",fg="white")
w.place(x=40,y=40)
w = Button(window,text="박스 #4",bg="green",fg="white")
w.place(x=60,y=60)
w = Button(window,text="박스 #5",bg="blue",fg="white")
w.place(x=80,y=80)


window.mainloop()

#요거