import sys
import random

while True:
  name = input("이름 >")
  if name =="" :
    sys.exit()

  q = input("질문 사항 >")
  print(name,"님",q,"에 대해 질문 주셨군요")

  an = random.randint(1,8)
 

  if an ==1 :
    print("아주 좋습니다.")

  if an ==2 :
    print("좋습니다.")

  if an ==3 :
    print("괜찮습니다.")

  if an ==4 :
    print("별롭니다.")

  if an ==5 :
    print("안 좋습니다.")

  if an ==6 :
    print("아주 안 좋습니다.")

  if an ==7 :
    print("하지마세요.")

  if an ==8 :
    print("최악 입니다. ")