import time

# 2부터 100까지 사이의 짝수 출력 

for i in range(2,101):
  if i %2 ==0:
    print(i)
  else:
    continue

#피티 10회씩 5세트  1세트 1회 입니다. ~ 1세트 10회 입니다. ~2세트 1회 입니다. 

for s in range(1,6):
  for h in range(1,11):
    print(f"{s}세트 {h}회 입니다.")
    time.sleep(1)