def countVowel(string):
  count = 0
  for ch in string:
    if ch in ['a','e','i','o','u']:
      count+=1
  return count

s = input("입력>")
n = countVowel(s)
print(f"모음의 개수는 {n}개 입니다.")
#모음의 개수 + 자음의 개수 