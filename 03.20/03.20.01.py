name = input("이름을 입력하시오: ")
age = int(input("나이를 입력하시오: "))
age+=1
print(name,"씨는 내년에 ",age,"살 이시네요 !")


where = str(input("주소:"))
many = int(input("방의 개수:"))
price = int(input("방의 개수:"))
print("주소: 서울시 종로구 \n방의 개수 : 3\n가격 : 100,000,000")
print("{}에 위치한 아주 좋은 아파트가 매물로 나왔습니다. 이 아파트는 {}개의 방을 가지고 있으며 가격은 {}입니다.".format(where,many,price))


one =float(input("첫 번째 숫자를 입력하시오:"))
two = float(input("두 번째 숫자를 입력하시오:"))
three = float(input("세 번째 숫자를 입력하시오:"))
print(one,two,three,"의 평균은",(one+two+three)/3,"입니다.")