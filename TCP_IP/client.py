import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect(('192.168.2.112', 9999))
s.connect(('127.0.0.1', 9999))
dir0 = '{"id"=1,"name"="ld","password"=123,"command"="reset"}'
dir1 = '{"id"=1,"name"="ld","password"=123,"command"="left"}'
dir2 = '{"id"=1,"name"="ld","password"=123,"command"="right"}'
dir3 = '{"id"=1,"name"="ld","password"=123,"command"="stop"}'
dir_pool=  [dir0, dir1, dir2, dir3]

#	data = s.recv(1024)
#	print 'server: '+data
#	dt = raw_input()
#	if dt=='exit':
#	break
#if __name__=='__main__':
while True:
#	data = s.recv(1024)
#	print 'server:'  + data
 	for dt in dir_pool:
		s.sendall(dt)

s.close()




