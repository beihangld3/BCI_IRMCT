import threading
import socket
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#s.bind(("192.168.2.112", 9999))
s.bind(("192.168.2.119", 9999))

s.listen(1)
print('Waiting for connection')
sock, addr = s.accept()
print 'Accept new connection from %s:%s' % addr
sock.send('Welcome!')
while True:
    data = sock.recv(1024)
    time.sleep(1)
    if data == 'exit' or not data:
        break
    print 'client: ' + data
    sock.send('%s received' % data)
sock.close()
print 'Connection from %s:%s closed' % addr

