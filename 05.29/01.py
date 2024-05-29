letters = ['A','B','C','D','E']

print(len(letters))

if 'A' in letters:
  print("True")

myList = ["영","일","이","삼"]

myList[1] = "추가"
print(myList)

myList.append("사")
print(myList)

myList.insert(1,"수정")
print(myList)

myList.remove("추가")
print(myList)

myList.pop(0)
print(myList)

if "수정" in myList:
  myList.remove("수정")
print(myList)

print(myList[0:2])
print(myList[:2])
print(myList[:])

number = [1,25,2,7,32,9,8]
number.sort()
print(number)
new_list= sorted(number,reverse=True)
print(new_list)