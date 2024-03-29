
height = float(input("키를 입력하세요.(m) :"))
weight = float(input("몸무게를 입력하세요. :"))

BMI = weight / (height * height)
print("당신의 BIM =",BMI)

height = float(input("키를 입력하세요.(cm) :"))/100
weight = float(input("몸무게를 입력하세요. :"))

BMI = weight / (height **2)
print("당신의 BIM =",BMI)

if BMI >= 25:
    print("비만입니다.")
elif BMI >= 23 and BMI < 25:
    print("과체중입니다.")
elif BMI >= 18.5 and BMI < 23:
    print("정상체중입니다.")
else:
    print("저체중입니다.")