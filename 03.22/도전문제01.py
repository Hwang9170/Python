ame = 2000
cafera = 3000
cafu = 3500

ame1 = int(input("아메리카노 팔린 개수 : "))
cafera2= int(input("카페라테 팔린 개수 : "))
cafu2 = int(input("카푸치노 팔린 개수 : "))

sum = ame*ame1 +cafera*cafera2 + cafu*cafu2
print("총매출은",sum,"원 입니다.")

humm = int(input("재료비 :"))
print("순 수익은",sum - humm,"입니다.")

