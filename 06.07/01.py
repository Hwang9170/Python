from tkinter import*

def process():
  tem = float(e1.get())
  mytemp = (tem-31)*5/9
  e2.insert(0,str(mytemp))
def process2():
  tem = float(e2.get())
  mytemp = (tem*9/5)+31
  e1.insert(0,str(mytemp))

window = Tk()

l1 =Label(window,text="화씨",font='helvetica 16 italic')
l2 =Label(window,text="섭씨",font='helvetica 16 italic')
l1.grid(row=0, column=0)
l2.grid(row=1, column=0)

e1 = Entry(window,bg= 'yellow',fg='black')
e2 = Entry(window,bg= 'yellow',fg='black')
e1.grid(row=0, column=0)
e2.grid(row=1, column=0)

b1 = Button(window, text="화씨 -> 섭씨",command=process)
b2 = Button(window,text="섭씨 -> 화씨",command=process2)
b1.grid(row=2,column=0)
b2.grid(row=2,column=1)

window.mainloop()


