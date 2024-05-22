def countVowel(string):
  count = 0
  for ch in string:
    if ch in ['a','e','i','o','u']:
      count+=1
  return count

def countConsonant(string):
  count = 0
  for ch in string:
    if ch in ['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z']:
      count+=1
  return count


s = input("입력>")
n1 = countVowel(s)
n2 = countConsonant(s)
print(f"모음의 개수는 {n1}개 입니다.")
print(f"자음의 개수는 {n2}개 입니다.")