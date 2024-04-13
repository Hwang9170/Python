name = input("이름>")
number = input("주민번호>")

y = number[0:2]
m = number[2:4]
d = number[4:6]
z = number[-1]
print(name,"님은",number[0:2],"년도",number[2:4],"월",number[4:6],"일에 태어났고","성별은",number[-1],"입니다.")
print(name,"님은",y,"년도",m,"월",d,"일에 태어났고 성별은",z,"입니다.")

if z == 1 or 3:
  z = "남자"
else:
  z = "여자"
print("{}님은 {}년 {}월 {}일에 태어났고 성별은 {}입니다.".format(name,y,m,d,z))