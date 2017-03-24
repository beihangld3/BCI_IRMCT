import socket

from pytpstreamer import TPStreamer

class TPInterface:

    AsServer = 0
    AsClient = 1

    Unsuccessful = -1
    Successful = 0

    ErrorSocket = 1
    ErrorEndpoint = 2
    ErrorBound = 3
    ErrorGeneric = 4
    ErrorNotSupported = 5
    ErrorProtocol = 6
    ErrorTimeout = 7

    def __init__(self):
        self.__socket = None
        self.__endpoint = None
        self._com = None
        self._stream = TPStreamer()
        self.__cache = ""

    def Plug(self, ip, port, mode):
        if mode == TPInterface.AsServer:
            return self.ConfAsServer(ip, port)
        elif mode == TPInterface.AsClient:
            return self.ConfAsClient(ip, port)
        else:
            return TPInterface.ErrorGeneric

    def Unplug(self):
        if self.__socket == None:
            return 

        self.__socket.close()
        self.__socket = None

        self._com = None

        if self.__endpoint:
            self.__endpoint.close()
            self.__endpoint = None

    def ConfAsServer(self, ip, port):
        if self.__socket:
            return TPInterface.ErrorBound

        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        try:
            self.__socket.bind((ip, int(port)))
            self.__socket.listen(1)
        except socket.error:
            self.Unplug()
            return TPInterface.ErrorSocket

        try:
            (self.__endpoint, addr) = self.__socket.accept()
        except socket.error:
            return TPInterface.ErrorEndpoint

        self._com = self.__endpoint
        return TPInterface.Successful

    def ConfAsClient(self, ip, port):
        if self.__socket:
            return TPInterface.ErrorBound

        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        try:
            self.__socket.connect((ip, int(port)))
        except socket.error:
            self.Unplug()
            return TPInterface.ErrorSocket

        self._com = self.__socket
        return TPInterface.Successful

    def IsPlugged(self):
        if not self._com:
            return False

        # TODO check if still connected
        return True

if __name__ == "__main__":
    import sys
    ti = TPInterface()
    if sys.argv[1] == "server":
        print 'ConfAsServer...',
        ret = ti.ConfAsServer("127.0.0.1", "3456")
        print 'Completed (%d)' % ret
    else:
        print 'ConfAsClient...',
        ret = ti.ConfAsClient("127.0.0.1", "3456")
        print 'Completed (%d)' % ret
