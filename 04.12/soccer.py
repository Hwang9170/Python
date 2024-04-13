import random
options=["왼쪽상단","왼쪽하단","중앙","오른쪽상단","오른쪽하단"]
computer_choice = random.choice(options)
user_choice = input("어디를 수비하시겠어요?(왼쪽상단,왼쪽하단, 중앙, 오른쪽상단,오른쪽하단) > ")
if computer_choice == user_choice: 
  print("수비에 성공하셨습니다. ")
else: 
  print("페널티킥이 성공하였습니다. ")
