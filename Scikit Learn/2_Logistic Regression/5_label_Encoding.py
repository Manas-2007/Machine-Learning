from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
departments=["IT","HR","Finance","AI","IT","IT","Finance"]
encoded_dept=encoder.fit_transform(departments)
print(encoded_dept)

reverse_dep=encoder.inverse_transform(encoded_dept)
print(reverse_dep)