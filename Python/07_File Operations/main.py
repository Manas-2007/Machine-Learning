# opening a file and reading data
file=open(r"C:\Users\Student\Desktop\Pandas\Machine-Learning\Python\07_File Operations\demo.txt","r")
data=file.read()
print("File Content :\n",data)

# Creating a new file writing data or overrides existing file
file=open(r"C:\Users\Student\Desktop\Pandas\Machine-Learning\Python\07_File Operations\demo.txt","w")
file.write("This is the VS code file WRITE function")

# appending some data at the end of the existing one
file=open(r"C:\Users\Student\Desktop\Pandas\Machine-Learning\Python\07_File Operations\demo.txt","a")
file.write("\nHello append function")

# Writing in a new file (auto created)
file=open("Write.txt","w")
file.write("This file is auto-created by write function")
