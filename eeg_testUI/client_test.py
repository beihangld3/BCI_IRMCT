host= '192.168.2.112'
port = 9999
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
s.sendall('{"id"=1,"name"="ld","password"=123,"command"="reset"}')
s.close()