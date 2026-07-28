amazon={
    "Product_ID":101,
    "Product" : "Watch",
    "Price":25000,
    "Brand":"Lexus"
}
print("Original Object :\n",amazon)

# To add new key-value pair
amazon["Discount"]=2000
print("\nAfter discount :\n",amazon)

# To remove some key-value pair (using .pop(key_name))
amazon.pop("Brand")
print("\nAfter removing brand :\n",amazon)

# To remove last key-value pair (using .popitem(key_name))
amazon.popitem()
print("\nAfter removing last pair :\n",amazon)

# Print only keys 
print("\nDisplaying keys in the amazon :")
for keys in amazon:
    print(keys)

# Print key-value pairs
print("\nDisplaying key-value pairs :")
for keys,values in amazon.items():
    print(f"{keys} : {values}")