# pylibtobicore.py -
# Copyright (C) 2011 Andrew Ramsay 
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# 	This is a Python port of the "tobicore" library from EPFL.

import time, math, re

# 
#   TCTypes
# 
TCSTATUS_VERSION        = "0.1.0.0"
TCSTATUS_ROOTNODE       = "tcstatus"
TCSTATUS_VERSIONNODE    = "version"
TCSTATUS_COMPONENTNODE  = "component"
TCSTATUS_STATUSNODE     = "status"

#
#   TCTimeval (meant to be equivalent to struct timeval in C)
#
class TCTimeval:
    def __init__(self):
        self.sec = 0
        self.usec = 0

    def clear(self):
        self.sec = 0
        self.usec = 0

    def isset(self):
        return (self.sec != 0 or self.usec != 0)

    def fromgettimeofday(self, v):
        foo = math.modf(v)
        self.sec = int(foo[1])
        self.usec = int(foo[0] * 1e06)

    def assign(self, s):
        (secs, usecs) = s.split(",")
        self.sec = int(secs)
        self.usec = int(usecs) * 1e06

#
#   TCTimestamp
#
class TCTimestamp:
    def __init__(self):
        self.Unset()

    def Unset(self):
        self.__timestamp = TCTimeval()

    def IsSet(self):
        return self.__timestamp.isset()

    def Tic(self):
        self.__timestamp.fromgettimeofday(time.time())

    def Toc(self, timestamp=None):
        if not timestamp:
            return self.Toc(self.__timestamp)

        toc = TCTimeval()
        toc.fromgettimeofday(time.time())

        return float((toc.sec - timestamp.sec)*1000000 + toc.usec - timestamp.usec)/1000.0

    def Get(self):
        cache = '%ld,%ld' % (self.__timestamp.sec, self.__timestamp.usec)
        self.__timestamp.clear()
        self.__timestamp.assign(cache)
        return cache
        
    def Set(self, timestamp):
        if isinstance(timestamp, TCTimeval):
            self.__timestamp.sec = timestamp.sec
            self.__timestamp.usec = timestamp.usec
        elif isinstance(timestamp, basestring):
            (self.__timestamp.sec, self.__timestamp.usec) = map(int, timestamp.split(","))
        else:
            # TODO error/exception
            print '[ERROR] TCTimestamp.Set() incorrect data type!', type(timestamp)
            return

# 
#   TCException
# 
class TCException(Exception):
    def __init__(self, info, caller = "undef"):
        Exception.__init__(self, info, caller)
        self.__info = info
        self.__caller = caller

    def GetCaller(self):
        return self.__caller

    def GetInfo(self):
        return self.__info

    def __str__(self):
        return "[%s] %s" % (self.__caller, self.__info)

    def __eq__(self, right):
        return (self.__info == right.GetInfo())

    def __ne__(self, right):
        return (self.__info != right.GetInfo())

#
#   TCBlock
#
class TCBlock:
    BlockIdxUnset = -1;

    def __init__(self):
        self.absolute = TCTimestamp()
        self.relative = TCTimestamp()
        self.__blockidx = TCBlock.BlockIdxUnset

    def SetBlockIdx(self, fidx):
        self.__blockidx = fidx
        return self.__blockidx

    def GetBlockIdx(self):
        return self.__blockidx

    def IncBlockIdx(self):
        self.__blockidx += 1
        return self.__blockidx

    def UnsetBlockIdx(self):
        self.__blockidx = TCBlock.UnsetBlockIdx

    def IsSetBlockIdx(self):
        return (self.__blockidx >= 0)

#
#   TCLanguage
# 
class TCLanguage:
    IA                  = 1
    IB                  = 2
    IC                  = 3
    ID                  = 4

    Ready               = 1
    Quit                = 2
    ErrorGeneric        = -1

    def __init__(self):
        pass

    def Status(self, component, status):
        return '<%s %s="%s" %s="%s" %s=%s/>' % (TCSTATUS_ROOTNODE, TCSTATUS_VERSIONNODE, TCSTATUS_VERSION, TCSTATUS_COMPONENTNODE, component, TCSTATUS_STATUSNODE, status)

    def CheckVersion(self, message):
        if message.startswith("<") and message.endswith("/>") and message.find("version") >= 0:
            idx = message.find("version")
            message = message[idx+9:]
            version = message[:message.find('"')]
            return (version == TCSTATUS_VERSION)

        return False

    def IsStatus(self, message):
        res = re.search('<tcstatus \S+ component="(\d+)" status="(\d+)" frame="(\d+)"/>', message)
        
        if len(res.groups()) != 3:
            return (False, TCLanguage.ErrorGeneric, TCLanguage.ErrorGeneric, TCLanguage.ErrorGeneric)

        return (True, res.group(1), res.group(2), res.group(3))

    def GetComponent(self, component):
        pass # (TODO: not implemented???)

    def GetStatus(self, component):
        pass # (TODO: not implemented???)

#
#   TCSerializer
#
class TCSerializer:
    def __init__(self, block = None):
        self.__message = block

    def SetMessage(self, block):
        self.__message = block

    # Abstract
    def Serialize(self, buffer):
        pass

    # Abstract
    def Deserialize(self, buffer):
        pass

    def SerializeCh(self, buffer):
        if not buffer:
            return None

        return self.Serialize(buffer)

    def DeserializeCh(self, buffer):
        if not buffer:
            return None

        return self.Deserialize(buffer)

