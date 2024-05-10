import random
import sys

s = input("패스워드 입력>")
ps = ['a','b','c','d','e','f',
      'g','h','i','j','k','l','m','n','o','p','q','r',
      's','t','u','v','w','x','y','z']
n = ['0','1','2','3','4','5','6','7','8','9']

found = False  # 패스워드가 발견되었는지 여부를 추적합니다.

for l in ps:
    for l2 in ps:
        for l3 in ps:
            for l4 in n:
                g = l + l2 + l3 + l4
                print(g)
                if g == s:
                    print("당신의 패스워드는 " + g)
                    found = True
                    break  # 내부 루프를 중단합니다.
            if found:
                break  # 외부 루프를 중단합니다.
        if found:
            break
    if found:
        break
