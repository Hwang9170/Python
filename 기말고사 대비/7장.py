import random
def print_add(name):
  print("서울시~~~")
  print(name)

print_add("황광호")

def get_sum(s,e):
  sum = 0
  for i in range(s,e+1):
    sum+=i
  return sum
print(get_sum(1,10))

def mo(string):
  count = 0
  for ch in string:
    if ch in ['a','e','i','o','u']:
      count +=1
  return count
s = input("문자열 입력:")
n = mo(s)
print(f"모음 개수는{n}개")


def lotto():
  number = []
  while len(number) < 6:
    s = random.randint(1,45)
    if s not in number:
      number.append(s)
  return number
print(f"생성된 로또 번호{lotto()}")

def great(name="철수"):
  print("안녕"+name+"님")
great()
great("민수")

def Max(a,b,c=-1000):
  if(a>= b) and (a>=c):
    L = a
  elif(b>=a)and(b>=c):
    L = b
  else:
    L=c
  return L
print(f"10,20,30 중에서 가장 큰 건 {Max(10,20,30)}")

