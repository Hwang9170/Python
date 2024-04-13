m= input("이번달은 몇월 달 입니까? : ")

if m == ("3" or "4" or "5" or "3월" or"4월"or"5월"):
 print("지금은 봄입니다.")
elif m == ("6" or "7" or "8" or "6월" or"7월"or"8월"):
  print("지금은 여름입니다.")
elif  m == ("9" or "10" or "11" or "9월" or"10월"or"11월"):
  print("지금은 가을입니다.")
elif  m ==("12" or "1" or "2" or "12월" or"1월"or"2월"):
  print("지금은 겨울입니다.")
else:
  print("올바른 입력이 아닙니다. 다시 입력바랍니다.")

h = 100
b = 3/5
i = 1
while i <=10:
 h  = h*b
 print(i,round(h,4))
 i = i + 1