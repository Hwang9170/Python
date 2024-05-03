import random

num_ran = random.randint(0,100)
guess = 0
num_in = 0

while guess<10:
    num_in = int(input("숫자를 입력하세요 :"))
    guess += 1
    if num_ran < num_in:
        print("랜덤 값보다 높음!")
    elif num_ran > num_in:
        print("랜덤 값보다 낮음!")
    else:
        print("축하합니다. 정답입니다.")
        break
print(guess, "번 숫자를 모두 입력했습니다.")