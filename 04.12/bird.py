import random
time = random.randint(1,24)
print("좋은 아침입니다. 지금 시간은"+str(time)+"시 입니다.")
sunny = random.choice([True,False])

if sunny:
  print("현재 날씨가 화창합니다.")
else:
  print("현재 날씨가 화창하지 않습니다.")

if sunny and 6<= time <9 or 14<time<16:
  print("종달새가 노래합니다.")
else:
  print("종달새가 노래하지 않습니다.")
