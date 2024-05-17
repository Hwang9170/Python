import random

while True:
    password = random.randint(0, 9999)
    an = int(input("숫자형태의 네 자리 암호를 입력하시오: "))
    
    if an == password:
        print("암호 풀기에 성공하셨습니다.")
        an2 = input("새로운 암호를 생성하시겠습니까? (Y/N): ")
        if an2.upper() == "Y":
            continue
        elif an2.upper() == "N":
            break
        else:
            print("잘못된 입력입니다. 다시 입력바랍니다.")
    else:
        print("숫자를 잘못 입력하셨습니다.")
