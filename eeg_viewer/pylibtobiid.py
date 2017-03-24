from xml.dom.minidom import Document
from xml.dom.minidom import parseString
from inspect import stack
from pylibtobicore import *

IDTYPES_FAMILY_UNDEF                = "undef"
IDTYPES_FAMILY_BIOSIG               = "biosig"
IDTYPES_FAMILY_CUSTOM               = "custom"

IDMESSAGE_VERSION                   = "0.0.2.1"
IDMESSAGE_ROOTNODE                  = "tobiid"
IDMESSAGE_VERSIONNODE               = "version"
IDMESSAGE_FRAMENODE                 = "frame"
IDMESSAGE_DESCRIPTIONNODE           = "description"
IDMESSAGE_FAMILYNODE                = "family"
IDMESSAGE_EVENTNODE                 = "event"
IDMESSAGE_TIMESTAMPNODE             = "timestamp"
IDMESSAGE_REFERENCENODE             = "reference"


def current_function():
    return stack()[1][3]

#
# IDAsServer
#

class IDAsServer:
    def __init__(self):
        pass

#
# IDAsClient
#

class IDAsClient(TCBlock):
    BlockThis   = 0
    BlockPrev   = 1
    BlockNext   = 2
    BlockAll    = 3

    def __init__(self):
        self.__queue = []

    def Add(self, message, updatefidx = False):
        self.__queue.append(message)
        if updatefidx: 
            TCBlock.SetBlockIdx(self, message.GetBlockIdx())

    def Get(message, idftype, idevent, direction):
        if not message:
            raise TCException("iD message needs to be allocated", '%s.%s' % (self.__class__.__name__, current_function()))

        if direction != IDAsClient.BlockAll and not TCBlock.IsSetBlockIdx(self):
            raise TCException("Block number must be set for searching Prev/Next", '%s.%s' % (self.__class__.__name__, current_function()))
    
        if self.Size() == 0:
            return None

        t_blockidx = TCBlock.BlockIdxUnset
        fmatch, tmatch, ematch = False, False, False

        for i in range(len(self.__queue)):
            t_blockidx = self.__queue[i].GetBlockIdx()
            t_type = self.__queue[i].GetFamilyType()
            t_event = self.__queue[i].GetEvent()

            # Match frame
            if direction == IDAsClient.BlockThis:
                fmatch = (t_blockidx == TCBlock.GetBlockIdx(self))
            elif direction == IDAsClient.BlockPrev:
                fmatch = (t_blockidx > TCBlock.GetBlockIdx(self))
            elif direction == IDAsClient.BlockNext:
                fmatch = (t_blockidx < TCBlock.GetBlockIdx(self))
            else: # BlockAll
                fmatch = True
        
            
            # Match type
            if idftype == IDMessage.FamilyBiosig:
                tmatch = (idftype == t_type)
            else: # IDMessage.FamilyUndef
                tmatch = True

            # Match event
            if idevent == IDMessage.EventNull:
                ematch = True
            else:
                ematch = (idevent == t_event)

            if tmatch and ematch and fmatch:
                m = self.__queue.pop(i)
                return m

        return None

    def Size(self):
        return len(self.__queue)

    def Clear(self):
        self.__queue = []
        return self.Size()

    def Dump(self):
        for i in self.__queue:
            i.Dump()

# 
# IDMessage
#

class IDMessage(TCBlock):

    FamilyUndef         = -1
    FamilyBiosig        = 0
    FamilyCustom        = 1

    EventNull           = -1

    TxtFamilyUndef      = "FamilyUndef"
    TxtFamilyBiosig     = "FamilyBiosig"
    TxtFamilyCustom     = "FamilyCustom"

    def __init__(self, other = None, familyType = None, event = None):
        TCBlock.__init__(self)
        TCBlock.SetBlockIdx(self, -1)
        self.__Init()
        if other:
            self.Copy(other)
        elif familyType and event:
            self.__familyType = familyType
            self.__event = event
            self.__description = "unset"

    def __Init(self):
        self.__familyType = IDMessage.FamilyUndef
        self.__event = IDMessage.EventNull
        self.__description = "unset"

    def Copy(self, other):
        TCBlock.SetBlockIdx(self, other.GetBlockIdx())
        self.__event = other.GetEvent()
        self.__familyType = other.GetFamilyType()
        self.__description = other.GetDescription()

    def GetDescription(self):
        return self.__description

    def SetDescription(self, description):
        self.__description = description

    def GetFamily(self):
        if self.__familyType == IDMessage.FamilyBiosig:
            return IDTYPES_FAMILY_BIOSIG
        elif self.__familyType == IDMessage.FamilyCustom:
            return IDTYPES_FAMILY_CUSTOM
        
        return IDTYPES_FAMILY_UNDEF

    def SetFamilyType(self, ftype):
        if isinstance(ftype, int):
            if ftype < IDMessage.FamilyUndef or ftype > IDMessage.FamilyCustom:
                return False
            self.__familyType = ftype
            return True
        else:
            if ftype == IDMessage.TxtFamilyUndef:
                self.__familyType = IDMessage.FamilyUndef
            elif ftype == IDMessage.TxtFamilyBiosig:
                self.__familyType = IDMessage.FamilyBiosig
            elif ftype == IDMessage.TxtFamilyCustom:
                self.__familyType = IDMessage.FamilyCustom
            else:
                return False
            return True

    def GetFamilyType(self):
        return self.__familyType

    def SetEvent(self, idevent):
        self.__event = idevent

    def GetEvent(self):
        return self.__event

    def Dump(self):
        print "[IDMessage::Dump] TOBI iD message for frame %d [%s]" % (TCBlock.GetBlockIdx(self), self.GetDescription())
        print " + Event family  %d/%s" % (self.GetFamilyType(), self.GetFamily())
        print " + Event value   %d" % (self.GetEvent())

    @staticmethod
    def FamilyType(family):
        if family == IDTYPES_FAMILY_BIOSIG:
            return IDMessage.FamilyBiosig
        else:
            return IDMessage.FamilyUndef
    
#
# IDSerializer
#

class IDSerializer:

    def __init__(self, message = None, indent = False, declaration = False):
        self.message = message or None
        self.__indent = indent
        self.__declaration = declaration

    def SetMessage(self, message):
        self.message = message

    def Serialize(self):
        if not self.message:
            raise TCException("iD message not sent, cannot serialize", '%s.%s' % (self.__class__.__name__, current_function()))

        doc = Document()

        # TODO declaration bit

        cacheFidx = '%d' % self.message.GetBlockIdx()
        cacheEvent = '%d' % self.message.GetEvent()
        fvalue = self.message.GetFamily()

        self.message.absolute.Tic()
        self.message.relative.Tic()
        timestamp = self.message.absolute.Get()
        reference = self.message.relative.Get()

        root = doc.createElement(IDMESSAGE_ROOTNODE)
        root.setAttribute(IDMESSAGE_VERSIONNODE, IDMESSAGE_VERSION)
        root.setAttribute(IDMESSAGE_DESCRIPTIONNODE, self.message.GetDescription())
        root.setAttribute(IDMESSAGE_FRAMENODE, cacheFidx)
        root.setAttribute(IDMESSAGE_FAMILYNODE, fvalue)
        root.setAttribute(IDMESSAGE_EVENTNODE, cacheEvent)
        root.setAttribute(IDMESSAGE_TIMESTAMPNODE, timestamp)
        root.setAttribute(IDMESSAGE_REFERENCENODE, reference)
        doc.appendChild(root)

        # TODO indent?

        return doc.toxml()

    def Deserialize(self, msg):
        try:
            doc = parseString(msg)
        except:
            #raise TCException("iD root note not found", '%s.%s' % (self.__class__.__name__, current_function()))
	    return False

        root = doc.documentElement

        if not root.hasAttribute(IDMESSAGE_VERSIONNODE) or root.getAttribute(IDMESSAGE_VERSIONNODE) != IDMESSAGE_VERSION:
            #raise TCException("iD version mismatch", '%s.%s' % (self.__class__.__name__, current_function()))
	    return False

        frame_number = int(root.getAttribute(IDMESSAGE_FRAMENODE))

        absolute = root.getAttribute(IDMESSAGE_TIMESTAMPNODE)
        self.message.absolute.Set(absolute)
        reference = root.getAttribute(IDMESSAGE_REFERENCENODE)
        self.message.relative.Set(reference)

        desc = root.getAttribute(IDMESSAGE_DESCRIPTIONNODE)
        self.message.SetDescription(desc)

        ft = root.getAttribute(IDMESSAGE_FAMILYNODE)
        if ft == IDTYPES_FAMILY_BIOSIG:
            self.message.SetFamilyType(IDMessage.FamilyBiosig)
        else:
            self.message.SetFamilyType(IDMessage.FamilyUndef)

        ev = root.getAttribute(IDMESSAGE_EVENTNODE)
        self.message.SetEvent(ev)

        return True
