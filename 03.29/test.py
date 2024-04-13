x = 10 
print("x = ", x)
x = 3.14
print("x = ", x)
x = "Hello Wolrd"
print("x = ", x)

print(type("안녕하세요"))
print(type(273))
print(type(3.5))

print("# 하나만 출력합니다.")
print("Hello Python Programming...!")
print()
print("# 여러개를 출력합니다.")
print(10,20,30,40,50)
print("안녕","하세요","저는","황광호입니다.")

print("안녕하세요")
print('안녕하세요')

print('"안녕"이라고 말했다.')

t = input("정수를 입력하시오: ")
x = int(t)
t = input("정수를 입력하시오: ")
y = int(t)
print(x+y)


print("나는 현재"+ str(20) +"살 입니다.")
print("원주율은 3.14 입니다.")

text = " Hello world "
print(text.count("o"))
print(text.upper())
print(text.lower())
print(text.swapcase())
print(text.title())
print(text.strip())
print(text.rstrip())
print(text.lstrip())
print(text.replace("world", "Hwang"))

ss = input("문자열 입력 ==> ")
print("출력 문자열 ==>", end ='')
for i in range(0,len(ss)):
  if ss[i]!= 'o':
    print(ss[i], end='')
  else:
    print('$',end='')

ss = "파이썬"

print(ss.center(10))
print(ss.center(10,'-'))
print(ss.ljust(10))
print(ss.rjust(10))
print(ss.zfill(10))




