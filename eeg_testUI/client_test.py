host= '127.0.0.1'
port = 9999
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
s.sendall('{"id"=1,"name"="ld","password"=123,"command"="right"}')
s.close()

