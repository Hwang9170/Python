
hi = "hello world!"
print(hi.upper()) #대문자로 변경
print(hi.lower()) #소문자로 변경
print(hi.swapcase()) #대문자는 소문자로 소문자는 대문자로 
print(hi.title()) #앞 글자들만 대문자로 

python = "   파   이   썬   "
python.strip()
python.rstrip() 
python.lstrip()
python.replace("파","카프리")

print(python.strip()) #양쪽 공백 제거 
print(python.rstrip()) #오른쪽 공백 제거
print( python.lstrip()) #왼쪽 공백 제거 
print( python.replace("파","카프리"))#파-> 카프리로 변경 

total = 0
for num in [10,20,30,40,50]:
  total =total+num
print("합계:",total)
print("----------")
total = 0
for num in [10,20,30,40,50]:
  total =total+num
  print("합계:",total)

total = 0
numbers = [10,20,30,40,50]
for num in numbers:
  total =total+num
print("합계:",total)

solar = ["수성","금성","지구","화성"]
count = 1
for star in solar:
  print("%s은 %d번째 행성"%(star,count))
  count = count+1






