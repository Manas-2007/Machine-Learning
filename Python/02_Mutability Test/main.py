list1=[1,2,3,4,5]
list2=[1,2,3,4,5]
print("Original List :\n",list1,"\n",list2)

# Test-1
print("\nSeparate Reference Test :")
list1[0]="Manas"
list2[2]="Ishanvi"
print(list1,"\n",list2)

# Test-2
list3=[1,2,3,4,5]
list4=list3
print("\nOriginal List :")
print(list3,"\n",list4)
list4[2]="Magic"
list3[0]="Manas"
print("\nSame Reference Test :")
print(list3,"\n",list4)

