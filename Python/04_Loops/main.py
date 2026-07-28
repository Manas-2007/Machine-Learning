listdata=[1,2,[3,4]]
listdata[2].append(5)
listdata.append(6)
print(listdata)


# List Comprehension way
even_list=[x for x in range(1,51) if(x%2==0)]
print("Even Number list :\n",even_list)


# Dictionary iterations
dictionary={
    "Roll_No":101,
    "Name":"Manas Patidar",
    "Branch" : "AIR",
    "CGPA" :9
}

# displaying only keys
for keys in dictionary:
    print(keys)

# displaying key-value pairs
print("\n\n")
for keys,values in dictionary.items():
    print(f"{keys} : {values}")

# Dictionary comprehension
cube_dict={x:x**3 for x in range(1,11)}
print("\n",cube_dict)


