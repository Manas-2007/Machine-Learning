# Normal Text function
def Display(x):
    print(f"Hello I am {x}, How are you!, how can I assist you......?")

# Even number display
def Even(n):
    for i in range(1,n+1):
        if(i%2==0):
            print(i, end=" , ")

# Even number display from LIST
def Even_list(list_name):
    new_list=[x for x in list_name if(x%2==0)]
    return new_list

