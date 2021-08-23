print("This is executed by API!")
f = open('material.py')
exec(f.read())
f.close()
