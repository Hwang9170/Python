m = 0
while True: 
  num = input('''메뉴판
1. 짜장 6000원
2. 짬뽕 7000원      
3. 볶음밥 8000원
4. 탕수육 20000원 
5. 주문 완료 
입력 바랍니다 >>''')
  if num == '1': 
    print("@자장면 1개 추가")
    m += 6000
  elif num =='2':
    print("@짬뽕 1개 추가")
    m += 7000
  elif num =='3':
    print("@볶음밥 1개 추가")
    m += 8000
  elif num =='4' : 
    print("@탕수육 1개 추가")
    m += 20000
  elif num =='5' :
    print("@주문 완료 총 주문 가격 : %d" %m)
    break
  else:
    print("잘못된 입력입니다. 다시 입력 바랍니다.")