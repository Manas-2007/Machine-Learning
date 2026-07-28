import dummy as dm

# Using Display function 
name=input("Enter your Name :")
dm.Display(name)
print("\n\n")

# Using Even Number function
number=int(input("Enter any Number :"))
dm.Even(number)
print("\n\n")

# Print Even number from the list
list_data=[x for x in range(50)]
result=dm.Even_list(list_data)
print("Even Number List :\n",result)
