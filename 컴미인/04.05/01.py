m= [90,25,67,45,80]
for n in range(len(m)):
  if m[n] <60:
    continue
  print("%d번 학생 축하합니다. 합격입니다."%(n+1))
print()

for i in range(2,10): #2에서 9까지 반복 
  for j in range(1,10): #1에서 9까지 반복 
    print(i*j,end="/") #예시 2 * 1 / end는 줄 바꾸지 않고 연결 
  print()

name = input("이름을 입력하세요:")

c = 10 
while True: 
  m = int(input("돈 넣으시오:"))
  if m == 300:
    print("커피 줌")
    c = c-1
  elif m >300:
    print("거스름돈 : %d원"%(m-300))
    c = c-1
  else:
    print("남은 커피 수 : %d개"%c)
  if c ==0: 
    print("커피 없음")
    break

