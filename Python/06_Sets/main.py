data1={1,2,3,1,2,3,1,2,3,"Manas"}
data2={"Manas",1,2,"Suhana","Pooja"}

print("Data1 :",data1,"Length = ",len(data1))
print("Data2 :",data2,"Length = ",len(data2))

union=data1.union(data2)
intersection=data1.intersection(data2)
difference=data2-data1

print("\nUnion :",union)
print("Intersection :",intersection)
print("Difference :",difference)

# Empty set vs Empty Dictionary
empty_dict={}
empty_set=set()
print("\nEmpty Set : ",type(empty_set),"Length = ",len(empty_set))
print("Empty Dict : ",type(empty_dict),"Length = ",len(empty_dict))

# Adding element in set
empty_set.add("Manas")
empty_set.add("Pooja")
empty_set.add("Mayur")
empty_set.add("Sumit")
empty_set.add("Yashi")
print("\nAfter entry of new data :\n",empty_set)

# Removing data from the set
empty_set.remove("Manas")
print("\nAfter removing of data :\n",empty_set)
