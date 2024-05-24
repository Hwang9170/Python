"""def greet(name,msg ="별일 없죠?"):
  print("안녕",name,''+msg)

greet("영화")"""

def getMax(a,b=-10000,c=-10000):
  if(a>=b)and(b>=c):
    largest = a
  elif(b>=a)and (b>=c):
    largest=b
  else:
   largest = c
  return largest

print(f"(10,20,30)중에서 최대값:{getMax(10,20,30)}")
print(f"(10,20)중에서 최대 값:{getMax(10,20)}")
print(f"(10)중에서 최대 값:{getMax(10)} ")