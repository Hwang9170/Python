id = "ilovepython" 
s = input("아이디를 입력하시오: ")
if id ==s:
 password = input('패스워드를 입력하시오: ')
 if password == '12345678':
  print('환영합니다.')
 else:
  print('잘못된 패스워드입니다. ')
else :
 print('잘못된 아이디입니다.')
