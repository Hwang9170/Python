import turtle

radius = 50

turtle.up() #펜을 올린것 (그림 그리기 중지)
turtle.goto(0, 0) #좌표 이동 (0,0)으로 
turtle.down() # 펜을 내린 것 (그림 그리기 가능)
turtle.circle(radius) #반지름 50인 원 그리기 

radius += 10 # 반지름 10 증가 
turtle.up() # 그림 중지
turtle.goto(100, 0) # 좌표 이동 
turtle.down() # 펜 내리기 
turtle.circle(radius) #반지름 60짜리 원 그리기 

radius += 10 #10 증가 
turtle.up() #그리기 중지 
turtle.goto(200, 0) #좌표 이동 (200,0) 으로 
turtle.down() # 그리기 가능 
turtle.circle(radius) #원 그리기 반지름 70

turtle.done() #그리기 끝 
