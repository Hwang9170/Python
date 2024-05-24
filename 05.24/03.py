def Happy(name):
  print("Happy BirthDay to you!")
  print("Happy BirthDay to you!")
  print(f"Happy BirthDay, dear {name}")
  print("Happy BirthDay to you!")

name = input("이름을 입력하시오>>")
if name == "":
  name = "Kim"
Happy(name)
