import random
import math
number = random.randint(1,100)
print(number)

n = int(input("정수를 입력하시오 :"))
fact = 1
for i in range(1,n+1):
  fact = fact*i
print(f"{n}의 팩토리얼은{fact}이다.")
log = ""
while log != "hi":
  log = input("로그인>>")
print("로그인 성공")

n = ""
sum = 0
while n != "no":
  k = int(input("숫자를 입력하시오:"))
  sum = sum+k 
  n = input("계속?(yes/no)") 
print(f"합계는{sum}")


print("1부터 100사이의 숫자를 맞추시오.")
 
k = random.randint(1,100)
o = 0
total = 0
while k != o:
  o = int(input("숫자를 입력하시오"))
  if (k>o):
    print("입력하신 값보다 큽니다.")
    total +=1
  if (k<o):
    print("입력하신 값보다 작습니다.")
    total +=1
print(f"정답!! 총{total}회 시도하셨습니다.")


while True:
  q = random.randint(1,100)
  p = random.randint(1,100)
  an = int(input(f"{q}+{p}="))
  
  if (an==q+p):
    print("잘했어요")
    o = input("그만하려면 y>>")
    if o =="y":
      break
  else:
    print("다음번엔 잘 할 수 있죠?")
    o = input("그만하려면 y>>")
    if o =="y":
      break

b = ["호밀","위트","화이트"]
m = ["소","돼지","닭"]
v =["양배추","양상추","엉겅퀴"]
s = ["1소스","2소스","3소스"]
for be in b:
  for me in m:
    for ve in v:
      for se in s:
        print(be,me,ve,se)


