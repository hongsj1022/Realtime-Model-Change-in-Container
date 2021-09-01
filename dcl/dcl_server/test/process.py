print("This is executed by API!")
a = 1
f = open('/home/pi/dcl/dcl_server/test/material.py')
exec(f.read())
f.close()
