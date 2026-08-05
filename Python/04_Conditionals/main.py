# Test-1
def Ai_Score(score):
    if((score<0) or (score>1)):
        print("....Invalid Score.....")
    elif((score>=0) and (score<0.5)):
        print("....Low Confidence....")
    elif((score>=0.5) and (score<0.9)):
        print("....Moderate Confidence....")
    elif((score>=0.9) and (score<=1)):
        print("....High Confidence....")
      
print("Test-1: ")  
Ai_Score(0.95)
Ai_Score(0.6)
Ai_Score(0.2)
Ai_Score(1.5)

# Test-2
def Navigate(left,center,right):
    status
    if((left==True) and (center==True) and (right==True)):
        status="Emergency Stop"
    elif((left==False) and (center==False) and (right==False)):
        status="Move Forward"
    elif((center==True) and ((left==False) or (right==False))):
        status="Reverse"
    elif((left==True) and ((center==False) and (right==False))):
        status="Steer Right"
    elif((right==True) and ((center==False) and (left==False))):
        status="Steer Left"
 
 
print("\nTest-2:")
print(Navigate(False,False,False))
 