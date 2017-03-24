import time

from pytpinterface import TPInterface
from pylibtobiid import IDSerializer, IDMessage
from pylibtobicore import TCBlock, TCLanguage


class TPiD(TPInterface):

    def __init__(self):
        TPInterface.__init__(self)
        self.__lang = TCLanguage()

    def Set(self, serializer, bidx = TCBlock.BlockIdxUnset, abidx = TCBlock.BlockIdxUnset):
        if self._com == None:
            return (TPInterface.ErrorSocket, None)
        if not self.IsPlugged():
            return (TPInterface.ErrorSocket, None)

        serializer.message.SetBlockIdx(bidx)
        serializer.message.absolute.Tic()
        self.__cache = serializer.Serialize()

        if self._com.send(self.__cache) != len(self.__cache):
            return (TPInterface.ErrorSocket, None)

        count = 0
        while True:
            self.__cache = ""
            self.__cache = self._com.recv(2048)
            if self.__cache == None or len(self.__cache) <= 0:
                count += 1
                if count == 100:
                    return (TPInterface.ErrorTimeout, None)
                time.sleep(0.001) # TODO??
                continue
            
            self._stream.Append(self.__cache)
            self._cache = ""

            self.__cache = self._stream.Extract("<tcstatus", "/>")
            if self.__cache:
                break

        (result, comp, status, fidx) = self.__lang.IsStatus(self.__cache)
        if result:
            abidx = int(fidx)
            return (TPInterface.Successful, abidx)

        return (TPInterface.ErrorProtocol, None)

    def Get(self, serializer):
        if self._com == None:
            return TPInterface.ErrorSocket
        if not self.IsPlugged():
            return TPInterface.ErrorSocket

        self.__cache = ""
        self.__cache = self._com.recv(2048)

        if len(self.__cache) > 0:
            self._stream.Append(self.__cache)

        self.__cache = self._stream.Extract("<tobiid", "/>")
        if not self.__cache:
            return TPInterface.Unsuccessful

        serializer.Deserialize(self.__cache)
        return TPInterface.Successful

if __name__ == "__main__":
    import sys

    tpid = TPiD()
    if sys.argv[1] == "receiver":
        tpid.ConfAsClient('127.0.0.1', '4567')
        print 'TPiD: client connected'
        ids = IDSerializer(IDMessage())
        for i in range(10):
            result = tpid.Get(ids)
            if result == TPInterface.Successful:
                print 'Success'
                # fake response message
                tpid._com.send('<tcstatus version="0.1.0.0" component="4" status="1" frame="11491"/>')
            else:
                print 'Error (%d)' % result
    else:
        tpid.ConfAsServer('127.0.0.1', '4567')
        print 'TPiD: server connected'
        ids = IDSerializer(IDMessage())
        for i in range(10):
            (result, buf) = tpid.Set(ids, i)
            if result == TPInterface.Successful:
                print 'Success'
            else:
                print 'Error (%d)' % result

