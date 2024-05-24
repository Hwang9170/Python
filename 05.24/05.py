x = int(input("1번째 숫자>>"))
y = int(input("2번째 숫자>>"))

def number(x,y):
  print(f"1번 : {x} + 2번: {y} = ")
  all = x+y
  sum = int(input("답은? >> "))
  if sum == all:
    print("정답 입니다.") 
  else: 
    print("오답 입니다.") 

number(x,y)

#3