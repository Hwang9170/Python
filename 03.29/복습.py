ss = input("문자열 입력 ==> ")
print("출력 문자열 ==>", end ='')
for i in range(0,len(ss)):
  if ss[i]!= 'o':
    print(ss[i], end='')
  else:
    print('$',end='')

print("")

ss = "파이썬"

print(ss.center(10)) #파이썬이 중앙에 
print(ss.center(10,'-')) #중앙에 양 옆에 -표시 
print(ss.ljust(10)) # 왼쪽에 붙이기 
print(ss.rjust(10)) # 오른쪽에 붙이기 
print(ss.zfill(10)) # 10자 중에 남는 자리 0으로 채우기 
