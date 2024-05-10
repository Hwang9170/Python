#랜덤 문제
import random

while 1:
  x = random.randint(1,100)
  y = random.randint(1,100)
  print(x,"+",y,"=",end="")
  a = int(input())
  if a == x+y:
    print("정답 !")
  else:
    print("다시 생각해보시오")

#호밀빵
b = ["호밀빵","위트","화이트"]
m = ["미트볼","소시지","닭가슴살"]
v = ["양상추","토마토","오이"]
s =["마요네즈","허니 머스타드","칠리"]
for B in b:
  for M in m:
    for V in v:
      for S in s:
        print(f"{b}+{m}+{v}+{s} =",b+m+v+s)