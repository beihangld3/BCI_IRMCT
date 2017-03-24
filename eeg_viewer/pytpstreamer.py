import threading

class TPStreamer:

    Forward = 0
    Reverse = 1

    def __init__(self):
        self.__stream = ""
        self.__mtxstream = threading.Lock()

    def Append(self, buffer):
        self.__mtxstream.acquire()
        self.__stream += buffer
        self.__mtxstream.release()

    def Extract(self, hdr, trl, direction = Forward):
        self.__mtxstream.acquire()

        if len(self.__stream) == 0:
            self.__mtxstream.release()
            return None

        if not self.__ImplHas(hdr, trl, direction):
            self.__mtxstream.release()
            return None

        p_hdr, p_trl, delta = -1, -1, 0

        if direction == TPStreamer.Forward:
            p_hdr = self.__stream.find(hdr)
            p_trl = self.__stream.find(trl, p_hdr)
        else:
            p_hdr = self.__stream.rfind(hdr)
            p_trl = self.__stream.rfind(trl, p_hdr)

        delta = len(trl)

        if p_hdr == -1 or p_trl == -1:
            self.__mtxstream.release()
            return None

        if p_hdr >= p_trl:
            return None

        buffer = self.__stream[p_hdr : p_trl+ delta]
        self.__stream = self.__stream[:p_hdr] + self.__stream[p_trl+delta:]
        self.__mtxstream.release()
        return buffer

    def Has(self, hdr, trl, direction = Forward):
        self.__mtxstream.acquire()
        result = self.__ImplHas(hdr, trl, direction)
        self.__mtxstream.release()
        return result

    def Count(self, hdr):
        count = 0

        self.__mtxstream.acquire()
        if len(self.__stream) != 0:
            pos = 0
            while pos != -1:
                pos = self.__stream.find(hdr, pos)
                if pos != -1:
                    count += 1
                    pos += 3 # why 3?

        self.__mtxstream.release()

        return count

    def Dump(self):
        self.__mtxstream.acquire()
        print "[TPStreamer::Dump] ", self.__stream
        self.__mtxstream.release()

    def Size(self):
        self.__mtxstream.acquire()
        size = len(self.__stream)
        self.__mtxstream.release()
        return size

    def Clear(self):
        self.__mtxstream.acquire()
        self.__stream = ""
        self.__mtxstream.release()

    def __ImplHas(self, hdr, trl, direction):
        if len(self.__stream) == 0:
            return False

        p_hdr, p_trl = -1, -1

        if direction == TPStreamer.Forward:
            p_hdr = self.__stream.find(hdr)
            p_trl = self.__stream.find(trl, p_hdr)
        else:
            p_hdr = self.__stream.rfind(hdr)
            p_trl = self.__stream.rfind(trl, p_hdr)

        if p_hdr == -1 or p_trl == -1:
            return False

        if p_hdr >= p_trl:
            return False

        return True

if __name__ == "__main__":
    tps = TPStreamer()
    tps.Append("bar foo 123 12345 bla 123 bla")
    assert(tps.Has("bar", "foo"))
    assert(not tps.Has("hello", "there"))
    assert(tps.Count("bla") == 2)
    buf = tps.Extract("bar", "bla")
    assert(buf != None)
    assert(len(buf) == 21)
    print 'Extracted: %s' % buf
    tps.Dump()
    assert(tps.Size() == 8)
