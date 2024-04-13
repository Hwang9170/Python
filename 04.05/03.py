import random

print("주사위 게임을 시작합니다.")

dice = random.randrange(6)
dice = dice +1 
if dice == 1:
  print("주사위의 눈은 {} 입니다.".format(dice))
if dice == 2:
  print("주사위의 눈은 {} 입니다.".format(dice))
if dice == 3:
  print("주사위의 눈은 {} 입니다.".format(dice))
if dice == 4:
  print("주사위의 눈은 {} 입니다.".format(dice))
if dice == 5:
  print("주사위의 눈은 {} 입니다.".format(dice))
if dice == 6:
  print("주사위의 눈은 {} 입니다.".format(dice))
print("게임이 종료되었습니다.")