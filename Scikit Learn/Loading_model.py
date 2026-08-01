import pandas as pd
import joblib

model=joblib.load("Laptop_price.pkl")

# taking user input
ram=int(input("Enter the RAM size : "))
storage=int(input("Enter the main storage size : "))
processor_gen=int(input("Enter the processor generation : "))
battery=int(input("Enter battery backup : "))
test_data=pd.DataFrame({
     "RAM_GB": [ram],
        "Storage_GB": [storage],
        "Processor_Gen": [processor_gen],
        "Battery_Hours": [battery],
})

prediction=model.predict(test_data)
print("Estimated Price (in Rupees) : ",round(prediction[0],1))
