i = 1
sum_value = 0
limit = 10000

while sum_value <limit:
  print("{}번  ".format(i))
  sum_value = sum_value + i 
  i=i+1
  print("=",sum_value)  
print(i-1,"번 더하면 됩니다.")
if sum_value > limit:
  print("i=1에서 1씩 증가시켜서 더하면 {}번째에 {}이 됩니다.".format(i,sum_value))
  print(sum_value)

print("---------")
a =0
i = 1
while a <10000:
  a += i
  i+= 1
print(i-1,"번 더하면 됨")