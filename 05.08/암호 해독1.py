import random
import sys

s = input("패스워드 입력>")
ps = ['a','b','c','d','e','f',
      'g','h','i','j','k','l','m','n','o','p','q','r',
      's','t','u','v','w','x','y','z']


for l in ps :
  for l2 in ps:
    for l3 in ps:
      g = l+l2+l3
      print(g)
      if g ==s:
        print("당신의 패스워드는"+ g)
        sys.exit
